from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)
from application.system_info import get_gpu_memory_usage
from huggingface_hub import snapshot_download
from application.llm_handler import LlmHandler
import torch
import time
import logging
import sys
import os
import math
import random

logger = logging.getLogger(__name__)

class ExllamaV2Generator(LlmHandler):
    def __init__(self):
        super().__init__()
        self.loras = {}

    def update_config(self, config_data):
        current_config = self.model_config
        merged_config = {**current_config, **config_data}
        self.model_config = merged_config

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'llm')
        return is_valid, errors

    def get_token_count(self, input_text):
        input_ids = self.tokenizer.encode(input_text)
        return input_ids.shape[-1]
    
    def stream(self, generator, tokenizer, model, prompt, channel, incoming_headers, 
               outgoing_properties, stops, request, model_data, lora):        
        
        # setup stop conditions
        check_stop_token, stop_conditions = self.build_stop_conditions(stops)
                
        # get starting time
        begin_time = time.time()

        # tokenize the prompt            
        input_ids = tokenizer.encode(prompt)
        input_token_count = input_ids.shape[-1]
        
        # set max new tokens and other params        
        max_new_tokens, top_p, top_k, seed, temperature, stream_output, debug, stop_key, \
                    min_p, mirostat, mirostat_eta, mirostat_tau = self.load_config_settings(input_token_count, request)
        
        if debug:
            print('\033[94m')
            print(request)
            print(prompt)
            print('\033[0m')                     

        if check_stop_token:
            stop_conditions.append(tokenizer.eos_token_id)

        if seed != -1: random.seed(seed)
        generator.warmup()
        generator.set_stop_conditions(stop_conditions)        
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.min_p = min_p
        if mirostat != 0:
            settings.mirostat = True
            settings.mirostat_tau = mirostat_tau
            settings.mirostat_eta = mirostat_eta        

        #settings.token_repetition_penalty = 1.05
        socket_id = incoming_headers["socket_id"] if "socket_id" in incoming_headers else None

        if "start_response" in request and stream_output:
            channel.basic_publish(
                    exchange=incoming_headers['return_exchange'], 
                    routing_key=incoming_headers['return_routing_key'], 
                    body=request["start_response"], properties=outgoing_properties)

        generated_tokens = 0
        stop_generation_counter = 0
        generator.begin_stream(input_ids, settings, loras = lora)        
        response = ""
        while True:
            chunk, eos, _ = generator.stream()
            if eos: break

            generated_tokens += 1            
            stop_generation, stop_generation_counter = self.check_stop_generation(stop_generation_counter, 
                                    model_data["stop_generation_event"], model_data["stop_generation_filter"], socket_id)        
            if stop_generation:
                finish_reason = "abort"
                break                            

            if generated_tokens >= max_new_tokens: 
                finish_reason = 'length'
                break

            # send chunk to front end
            if stream_output:
                if debug:
                    print('\033[96m' + chunk, end="")
                    sys.stdout.flush()

                channel.basic_publish(
                    exchange=incoming_headers['return_exchange'], 
                    routing_key=incoming_headers['return_routing_key'], 
                    body=chunk, properties=outgoing_properties)
            else:
                response += chunk            

        if debug and stream_output:
            print('\033[0m' + "")
        
        finish_reason = "stop"
        end_time = time.time()
        elapsed = end_time - begin_time
        token_rate = 0 if elapsed == 0 else (generated_tokens / elapsed)        
        model_name = incoming_headers["model_name"] if "model_name" in incoming_headers else "not_provided"
        return self.finish_response(stop_key, response, request, stream_output, finish_reason, 
                                        token_rate, generated_tokens, input_token_count, model_name, elapsed, debug)


    def load_lora(self, request, model, config):

        # load lora from config and override w/ request if present
        lora_name = config["default_lora"] if "default_lora" in config else None
        if "lora" in request:
            lora_name = request["lora"]

        if lora_name != None:
            if lora_name not in self.loras:

                logger.info(f"loading lora {lora_name}")
                lora_dir = os.path.join(f"data/loras/", lora_name)
                if not os.path.exists(lora_dir):
                    logger.info("downloading lora {lora_name} from huggingface")
                    snapshot_download(repo_id=lora_name, local_dir=lora_dir, cache_dir='data/cache', local_dir_use_symlinks=False)
                
                lora = ExLlamaV2Lora.from_directory(model["model_loaded"], lora_dir)
                self.loras[lora_name] = lora
            else:
                logger.info(f"using lora {lora_name}")

            return self.loras[lora_name]
        
        return None

    def execute(self, model, request):
        config = self.model_config        
        
        # build the prompt
        prompt = self.build_prompt(request, config, model)
        incoming_headers = model["amqp_headers"]
        outgoing_properties = self.copy_queue_headers(incoming_headers)

        # lora code
        lora = self.load_lora(request, model, self.model_config)

        # last string to send after done streaming output                        
        stream_resp = self.stream(
            model["generator"], 
            model["tokenizer"], 
            model["model_loaded"], 
            prompt,
            model["amqp_channel"],
            incoming_headers,
            outgoing_properties,
            config["stop_on"],
            request,
            model,
            lora)
        
        return stream_resp
        
    def load(self, model, model_options, local_path):           
        self.model_config = model["configuration"]              
        load_error = False
        try:
            model_path = local_path
            if "branch" in model["model"][0] and model_options["use_precision"] in model["model"][0]["branch"]:                
                branch_path = model["model"][0]["branch"][model_options["use_precision"]]
                model_path = f"{local_path}/{branch_path}"

            config = ExLlamaV2Config()
            config.model_dir = model_path
            config.prepare()

            if model_options["device"].startswith("split"):
                device_map = model_options["device"].split(':')[1].split(",")            
                device_map = list(map(int, device_map))
            elif model_options["device"].startswith("cuda"):
                device_number = int(model_options["device"].split(':')[1])
                device_array = [0]*12
                used_memory, free_memory, total_memory = get_gpu_memory_usage(device_number)
                device_array[device_number] = math.floor(total_memory / 1024)
                last_non_zero = len(device_array) - 1
                while last_non_zero > 0 and device_array[last_non_zero] == 0:
                    last_non_zero -= 1
                device_array = device_array[:last_non_zero + 1]
                device_map = device_array
                                        
            logger.info(f"starting module {model_path}")          
            model_loaded = ExLlamaV2(config)                                                     
            model_loaded.load(gpu_split=device_map)
            cache = ExLlamaV2Cache(model_loaded)
            tokenizer = ExLlamaV2Tokenizer(config)
            generator = ExLlamaV2StreamingGenerator(model_loaded, cache, tokenizer)   
            self.tokenizer = tokenizer         

            logger.info(f'skill {model["routing_key"]} loaded to {model_options["device"]}, precision: {model_options["use_precision"]}')
            return { "model_loaded": model_loaded, "generator": generator, "tokenizer": tokenizer, "error": load_error }                        
        except Exception as e:
            logger.error(f"error loading model")
            print(e)
            load_error = True
            return { "error": load_error }