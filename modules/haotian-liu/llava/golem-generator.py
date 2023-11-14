#from transformers_stream_generator import init_stream_support
#init_stream_support()
from application.llm_handler import LlmHandler
import requests
import torch
import json
import logging
from PIL import Image
from io import BytesIO
import os
import sys
import time
from transformers.generation.streamers import BaseStreamer

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

class AmqpStreamer(BaseStreamer):
    def __init__(self, tokenizer, channel, incoming_headers, outgoing_properties, model_data, check_function, debug):
        self.tokenizer = tokenizer                
        self.all_tokens = []
        self.all_text = ""
        self.new_tokens = 0            
        self.debug = debug
        self.channel = channel
        self.outgoing_properties = outgoing_properties
        self.incoming_headers = incoming_headers
        self.model_data = model_data
        self.finish_reason = "stop"
        self.check_stop_generation = check_function
        self.stop_generation_counter = 0
        self.socket_id = incoming_headers["socket_id"] if "socket_id" in incoming_headers else None

    def get_new_tokens(self):
        return self.new_tokens
    
    def get_response(self):
        return self.all_text
    
    def get_finish_reason(self):
        return self.finish_reason
    
    def put(self, value):        

        stop_generation, self.stop_generation_counter = self.check_stop_generation(self.stop_generation_counter, 
                self.model_data["stop_generation_event"], self.model_data["stop_generation_filter"], self.socket_id)
        
        if stop_generation:
            self.finish_reason = "abort"
            logger.info("stopping generation of text")
            raise ValueError("stopping generation of text")
        
        self.all_tokens.extend(value.tolist())
        new_text = self.tokenizer.decode(self.all_tokens)
        new_chunk = new_text[len(self.all_text):]
        self.all_text += new_chunk
        self.new_tokens += 1             

        if self.debug:
            print('\033[96m' + new_chunk, end="")

        self.channel.basic_publish(
            exchange=self.incoming_headers['return_exchange'], 
            routing_key=self.incoming_headers['return_routing_key'], 
            body=new_chunk, properties=self.outgoing_properties)

    def end(self):
        pass      

class LLaVA(LlmHandler):
    def __init__(self):
        super().__init__()

    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'llm')        
        return is_valid, errors

    def get_token_count(self, input_text):
        input_ids = tokenizer_image_token(input_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_token_count = input_ids.shape[1]
        print(f"INPUT: {input_text}\nTOKEN COUNT: {input_token_count}\n\n")
        return input_token_count
    
    def execute(self, model, request):
        
        stream_output = request.get("stream", False)
        image_found = request.get("img_url", None)
        if image_found and "messages" in request and request["messages"][-1]["role"] == "user":
            logger.info(f"loading image {image_found}")
            new_message = f"<image>\n{request['messages'][-1]['content']}"
            request["messages"][-1]["content"] = new_message
            image = self.load_image(request["img_url"])
            image_tensor = model["image_processor"].preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()        
        else:
            image_tensor = None
        
        config = self.model_config
        prompt = self.build_prompt(request, config, model)
        input_ids = tokenizer_image_token(prompt, model["tokenizer"], IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_token_count = input_ids.shape[1]

        finish_reason = "stop"
        max_new_tokens, top_p, top_k, seed, temperature, stream_output, debug, stop_key = self.load_config_settings(input_token_count, request)        
        check_stop_token, stop_conditions = self.build_stop_conditions(config["stop_on"])
        stopping_criteria = KeywordsStoppingCriteria(stop_conditions, model["tokenizer"], input_ids)

        if debug:
            print('\033[94m')
            print(json.dumps(request, indent=2))
            print(prompt)     
            print('\033[0m')        

        begin_time = time.time()                
        with torch.inference_mode():            

            do_sample = True if seed == -1 else False
            incoming_headers = model["amqp_headers"]
            outgoing_properties = self.copy_queue_headers(incoming_headers)

            if stream_output:                
                streamer = AmqpStreamer(model["tokenizer"], model["amqp_channel"], incoming_headers, outgoing_properties, model, self.check_stop_generation, debug)
                try:
                    output_ids = model["model_loaded"].generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        top_k=int(top_k * 100),
                        top_p=top_p,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])        
                except Exception as e:
                    print(e)
                    pass
            else:
                output_ids = model["model_loaded"].generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=do_sample,
                    top_k=int(top_k * 100),
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])  
        
        new_tokens = streamer.get_new_tokens() if stream_output == True else output_ids.shape[1] - input_token_count
        end_time = time.time()
        elapsed = end_time - begin_time
        token_rate = 0 if elapsed == 0 else (new_tokens / elapsed)

        response = model["tokenizer"].decode(output_ids[0, input_ids.shape[1]:]) if stream_output == False else ""
        model_name = incoming_headers["model_name"] if "model_name" in incoming_headers else "not_provided"
        return self.finish_response(stop_key, response, request, stream_output, finish_reason, 
                                        token_rate, new_tokens, input_token_count, model_name, elapsed, debug)        
        
    def load(self, model, model_options, local_path):
        self.config = model        
        self.model_config = model["configuration"]            
        # tried to used both of these but they do not work, 4-bit fails to load and 8-bit outputs random shit
        load_4bit = False
        load_8bit = True if model_options["use_precision"] == "8-bit" else False
        
        try:
            model_name = get_model_name_from_path(local_path)
            base_model = f"data/models/{model['model'][1]['name']}"
            tokenizer, model, image_processor, context_len = load_pretrained_model(local_path, base_model, model_name, load_8bit, load_4bit, device=model_options["device"])   
            self.tokenizer = tokenizer
            return {"model_loaded":model, "tokenizer": tokenizer, "image_processor": image_processor, "device": model_options["device"]}
        except Exception as e:
            logger.error(f"error loading model")
            load_error = True
            print(e)            
            return { "error": load_error }
