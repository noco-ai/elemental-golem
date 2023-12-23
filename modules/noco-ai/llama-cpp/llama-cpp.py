from llama_cpp import Llama
from application.llm_handler import LlmHandler
import torch
import time
import logging

logger = logging.getLogger(__name__)

class GGUFGenerator(LlmHandler):
    def __init__(self):
        super().__init__()
        self.loras = {}

    def update_config(self, config_data):
        current_config = self.model_config
        merged_config = {**current_config, **config_data}
        self.model_config = merged_config

    def get_token_count(self, input_text):
        inputs = self.loaded_model.tokenize(bytes(input_text, 'utf-8'))
        return len(inputs)

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'llm')
        return is_valid, errors

    def stream(self, model, prompt, channel, incoming_headers, 
               outgoing_properties, stops, request, model_data):
        
        # setup stop conditions
        check_stop_token, stop_conditions = self.build_stop_conditions(stops, False)
                
        # get starting time
        begin_time = time.time()        
        
        # set max new tokens and other params
        prompt_tokens = model.tokenize(bytes(prompt, 'utf-8'))
        input_token_count = len(prompt_tokens)
        max_new_tokens, top_p, top_k, seed, temperature, stream_output, debug, stop_key, \
                    min_p, mirostat, mirostat_eta, mirostat_tau = self.load_config_settings(input_token_count, request)
        if debug:
            print('\033[94m')
            print(request)
            print(prompt)
            print('\033[0m')                    
        
        response = ""
        new_tokens = 0
        finish_reason = 'stop'                            
        socket_id = incoming_headers["socket_id"] if "socket_id" in incoming_headers else None
        stop_generation_counter = 0
        model_args = {}

        # sampler settings
        if mirostat != 0:
            model_args["mirostat_mode"] = mirostat
            model_args["mirostat_eta"] = mirostat_eta
            model_args["mirostat_tau"] = mirostat_tau
        if seed != -1:
            model_args["seed"] = seed

        for model_stream in model(prompt, stream=True, max_tokens=max_new_tokens, min_p=min_p,
            temperature=temperature, stop=stop_conditions, top_k=top_k, top_p=top_p, **model_args):
            text = model_stream["choices"][0]["text"]
            
            stop_generation, stop_generation_counter = self.check_stop_generation(stop_generation_counter, 
                                    model_data["stop_generation_event"], model_data["stop_generation_filter"], socket_id)

            if stop_generation:
                finish_reason = "abort"
                break                            

            new_tokens += 1
            if new_tokens >= max_new_tokens: 
                finish_reason = 'length'
                break
            
            # send chunk to front end
            if stream_output:
                if debug:
                    print('\033[96m' + text, end="")

                channel.basic_publish(
                    exchange=incoming_headers['return_exchange'], 
                    routing_key=incoming_headers['return_routing_key'], 
                    body=text, properties=outgoing_properties)
            else:
                response += text

        if debug and stream_output:
            print('\033[0m' + "")

        end_time = time.time()
        elapsed = end_time - begin_time
        token_rate = 0 if elapsed == 0 else (new_tokens / elapsed)        
        model_name = incoming_headers["model_name"] if "model_name" in incoming_headers else "not_provided"
        return self.finish_response(stop_key, response, request, stream_output, finish_reason, 
                                        token_rate, new_tokens, input_token_count, model_name, elapsed, debug)

    def execute(self, model, request):
        config = self.model_config                    

        # build the prompt        
        prompt = self.build_prompt(request, config, model)
        incoming_headers = model["amqp_headers"]
        outgoing_properties = self.copy_queue_headers(incoming_headers)        

        # last string to send after done streaming output                        
        stream_resp = self.stream(            
            model["model_loaded"], 
            prompt,
            model["amqp_channel"],
            incoming_headers,
            outgoing_properties,
            config["stop_on"],
            request,
            model)
        
        return stream_resp
        
    def load(self, model, model_options, local_path):           
        self.model_config = model["configuration"]      
        
        try:                        
            if not model["model"][0]["files"][model_options["use_precision"]]:
                return { "error": True }

            lora_name = self.model_config["default_lora"] if "default_lora" in self.model_config else None
            model_file = model["model"][0]["files"][model_options["use_precision"]]
            model_path = f"{local_path}/{model_file}"
            config_threads = model["configuration"].get("num_threads", -1)
            num_threads = None if config_threads == -1 else config_threads
            max_seq_len = model["configuration"].get("max_seq_len", 2048)
            model_args = {
                "model_path": model_path, 
                "n_gpu_layers": 0, 
                "n_ctx": max_seq_len,
                "n_threads":num_threads
            }

            if lora_name != None:
                model_args["lora_path"] = f"data/loras/{lora_name}/"

            if model_options["device"].startswith("cuda"):
                model_args["n_gpu_layers"] = model["configuration"].get("model_layers", 0)
                model_args["main_gpu"] = int(model_options["device"].split(":")[1]) 
                #gpu_device = int(model_options["device"].split(":")[1])                
                #tensor_map = [0] * gpu_device + [1]
                #tensor_map[0] = 0.01 
                #tensor_split=tensor_map, 
                #tensor_map[gpu_device] = 0.99
                #print(tensor_map)
                #gpu_device = 0
            if "70b" in model_file:
                model_args["n_gqa"] = 8   

            load_model = Llama(**model_args)
            self.loaded_model = load_model
            
            print(f'skill {model["routing_key"]} loaded to {model_options["device"]}')
            return { "model_loaded": load_model, "error": False }             
        except Exception as e:
            print(f"error loading model")
            print(e)
            return { "error": True }