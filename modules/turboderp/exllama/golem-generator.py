from application.llm_handler import LlmHandler
import sys
import os
import glob
import time
import logging
import math
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from lora import ExLlamaLora
from application.system_info import get_gpu_memory_usage
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class GolemExLlamaGenerator(LlmHandler):
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
        ids = self.generator.tokenizer.encode(input_text)
        input_token_count = len(ids[0])
        return input_token_count
    
    def stream(self, generator, tokenizer, model, prompt, channel, incoming_headers, 
               outgoing_properties, stops, model_data, request):
        
        # setup stop conditions
        check_stop_token, stop_conditions = self.build_stop_conditions(stops)        
        
        res_line = ""        
        held_text = ""
        response = ""
        unicode_hold = False        
        finish_reason = "stop"
        stop_condition = False        
        new_tokens = 0
        stop_generation_counter = 0
        ids = generator.tokenizer.encode(prompt)
        input_token_count = len(ids[0])

        max_new_tokens, top_p, top_k, seed, temperature, stream_output, debug, stop_key, \
                    min_p, mirostat, mirostat_eta, mirostat_tau = self.load_config_settings(input_token_count, request)        
        
        if debug:
            print('\033[94m')
            print(request)
            print(prompt)
            print('\033[0m')                                     
        
        socket_id = incoming_headers["socket_id"] if "socket_id" in incoming_headers else None
        generator.settings.temperature = temperature
        generator.settings.top_p = top_p        
        begin_time = time.time()

        generator.gen_begin(ids)
        generator.begin_beam_search()
        for i in range(max_new_tokens):
            new_tokens += 1

            # check if stop generation was requested
            stop_generation, stop_generation_counter = self.check_stop_generation(stop_generation_counter, 
                                    model_data["stop_generation_event"], model_data["stop_generation_filter"], socket_id)
                
            if stop_generation:
                finish_reason = "abort"
                break                            

            token = generator.beam_search()
            prev_res_line = res_line
            res_line = tokenizer.decode(generator.sequence_actual[0, -new_tokens:])
            new_text = res_line[len(prev_res_line):]

            # new text
            chunk = held_text + new_text

            # check if we should hold off on streaming this text
            hold_text = False
            for stop_string in stop_conditions:
                if stop_string.startswith(chunk.lower()): hold_text = True
            
            if len(res_line): 
                check_ord = ord(res_line[-1])
                if check_ord == 65533 or check_ord == 55356 or check_ord == 55357:
                    hold_text = True
                    unicode_hold = True

            if not hold_text:
                if unicode_hold is True:
                    unicode_hold = False
                    chunk = res_line[-1:]

                # send chunk to front end
                if stream_output:
                    if debug:
                        print('\033[96m' + chunk, end="")
                        
                    channel.basic_publish(
                        exchange=incoming_headers['return_exchange'], 
                        routing_key=incoming_headers['return_routing_key'], 
                        body=chunk, properties=outgoing_properties)
                else:
                    response += chunk

                prompt += chunk
                held_text = ""
            else:
                held_text += new_text

            # check stop conditions                
            stop_condition = self.check_stop_conditions(token, res_line, tokenizer.eos_token_id, 
                                                    check_stop_token, stop_conditions)
            if stop_condition: break

        end_time = time.time()
        elapsed = end_time - begin_time
        token_rate = 0 if elapsed == 0 else (new_tokens / elapsed)
        generator.end_beam_search()
        
        if debug and stream_output:
            print('\033[0m' + "")

        if new_tokens == max_new_tokens:
            finish_reason = "length"

        model_name = incoming_headers["model_name"] if "model_name" in incoming_headers else "not_provided"
        resp = self.finish_response(stop_key, response, request, stream_output, finish_reason, 
                                        token_rate, new_tokens, input_token_count, model_name, elapsed, debug)                
        return resp
    
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
 
                lora_path = os.path.join(f"data/loras/", lora_name, "adapter_model.bin")                
                lora_config_path = os.path.join(f"data/loras/{lora_name}", "adapter_config.json")
                
                lora = ExLlamaLora(model["model_loaded"], lora_config_path, lora_path)
                self.loras[lora_name] = lora
            else:
                logger.info(f"using lora {lora_name}")

            model["generator"].lora = self.loras[lora_name]
        else:
            model["generator"].lora = None

    def execute(self, model, request):
        # load lora
        config = self.model_config        
        self.load_lora(request, model, config)

        # build prompt
        prompt = self.build_prompt(request, config, model)

        # copy amqp headers
        incoming_headers = model["amqp_headers"]
        outgoing_properties = self.copy_queue_headers(incoming_headers)

        stream_resp = self.stream(
            model["generator"], 
            model["tokenizer"], 
            model["model_loaded"], 
            prompt,
            model["amqp_channel"],
            incoming_headers,
            outgoing_properties,
            config["stop_on"],     
            model,       
            request)
        
        return stream_resp
        
    def load(self, model, model_options, local_path):        
        self.model_config = model["configuration"]     

        # get paths
        logger.info(f"starting module {local_path}")
        tokenizer_path = os.path.join(local_path, "tokenizer.model")
        model_config_path = os.path.join(local_path, "config.json")
        st_pattern = os.path.join(local_path, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        # Create config, model, tokenizer and generator
        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path
        config.compress_pos_emb = model["configuration"].get("compress_pos_emb", 1.0)
        config.max_seq_len = model["configuration"].get("max_seq_len", 2048)
        config.matmul_recons_thd = 8
        config.fused_mlp_thd = 2
        config.sdp_thd = 8

        # set model device        
        if model_options["device"].startswith("split"):
            device_map = model_options["device"].split(':')[1]
            config.set_auto_map(device_map)
        elif model_options["device"].startswith("cuda"):
            device_number = int(model_options["device"].split(':')[1])
            device_array = [0]*12
            used_memory, free_memory, total_memory = get_gpu_memory_usage(device_number)
            device_array[device_number] = math.floor(total_memory / 1024)
            last_non_zero = len(device_array) - 1
            while last_non_zero > 0 and device_array[last_non_zero] == 0:
                last_non_zero -= 1
            device_array = device_array[:last_non_zero + 1]
            device_map = ','.join(map(str, device_array))
            config.set_auto_map(device_map)        

        load_error = False
        try:
            load_model = ExLlama(config)                                            
            tokenizer = ExLlamaTokenizer(tokenizer_path)           
            cache = ExLlamaCache(load_model)                       
            generator = ExLlamaGenerator(load_model, tokenizer, cache)  

            # Configure generator         
            self.generator = generator   
            generator.settings.min_p = 0.0
            generator.settings.top_k = 0
            generator.settings.typical = 0.25
            generator.settings.token_repetition_penalty_max = 1.15
            generator.settings.token_repetition_penalty_sustain = 2048
            generator.settings.token_repetition_penalty_decay = 512
            
            logger.info(f'skill {model["routing_key"]} loaded to {model_options["device"]}')
            return { "model_loaded": load_model, "generator": generator, "tokenizer": tokenizer, "error": load_error }                        
        except Exception as e:
            logger.error(f"error loading model")
            load_error = True
            print(e)            
            return { "error": load_error }