from transformers_stream_generator import init_stream_support
init_stream_support()
from application.llm_handler import LlmHandler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import logging

logger = logging.getLogger(__name__)

class TransformersGenerator(LlmHandler):
    def __init__(self):
        super().__init__()

    def update_config(self, config_data):
        current_config = self.model_config
        merged_config = {**current_config, **config_data}
        self.model_config = merged_config

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'llm')
        return is_valid, errors

    def get_token_count(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        return inputs["input_ids"].shape[1]
    
    def stream(self, generator, tokenizer, model, prompt, channel, incoming_headers, 
               outgoing_properties, stops, request, model_data):        

        # setup stop conditions
        check_stop_token, stop_conditions = self.build_stop_conditions(stops)

        # force this to false, token passed to check_stop_conditions not same format as other handlers
        check_stop_token = False        
                
        # get starting time
        begin_time = time.time()

        # tokenize the prompt        
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        input_token_count = inputs["input_ids"].shape[1]
        
        # set max new tokens and other params
        max_new_tokens, top_p, top_k, seed, temperature, stream_output, debug, stop_key = self.load_config_settings(input_token_count, request)
        if debug:
            print('\033[94m')
            print(request)
            print(prompt)
            print('\033[0m')                                     

        generator = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            do_stream=True,
            top_p=top_p,
            top_k=int(top_k * 100),
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
        )

        # vars used in generation loop
        all_tokens = []
        all_text = ""
        response = ""
        held_text = ""
        new_tokens = 0            
        finish_reason = 'stop'
        socket_id = incoming_headers["socket_id"] if "socket_id" in incoming_headers else None
        stop_generation_counter = 0

        for token in generator:                
            all_tokens.extend(token.tolist())
            new_text = tokenizer.decode(all_tokens)
            new_chuck = new_text[len(all_text):]
            all_text += new_chuck
            new_tokens += 1                              

            if new_tokens >= max_new_tokens: 
                finish_reason = 'length'
                break

            stop_generation, stop_generation_counter = self.check_stop_generation(stop_generation_counter, 
                                    model_data["stop_generation_event"], model_data["stop_generation_filter"], socket_id)

            if stop_generation:
                finish_reason = "abort"
                break                            

            # check if we should hold off on streaming this text
            hold_text = False
            for stop_string in stop_conditions:                        
                if len(held_text) and stop_string.startswith(held_text.lower() + new_chuck.lower()): hold_text = True                            
                elif stop_string.startswith(new_chuck.lower()): hold_text = True                            

            if not hold_text:                    

                # send chunk to front end
                if stream_output:
                    if debug:
                        print('\033[96m' + new_chuck, end="")

                    channel.basic_publish(
                        exchange=incoming_headers['return_exchange'], 
                        routing_key=incoming_headers['return_routing_key'], 
                        body=new_chuck, properties=outgoing_properties)
                else:
                    response += new_chuck

                held_text = ""
            else:
                held_text += new_chuck                

            stop_condition = self.check_stop_conditions(token, held_text, tokenizer.eos_token_id, 
                                                        check_stop_token, stop_conditions)
            if stop_condition: break

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
            model["generator"], 
            model["tokenizer"], 
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

        # get paths
        logger.info(f"starting module {local_path}")        
        load_error = False
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            quantization_config = None            
            if model_options["use_precision"] != "full":
                if model_options["use_precision"] == "4-bit":                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )

            # this is not fully impelemented and should but a device map based off the split not auto
            device_map = "auto" if model_options["device"].startswith("split") else model_options["device"]
            load_model = AutoModelForCausalLM.from_pretrained(
                local_path,
                quantization_config=quantization_config,
                device_map=device_map
            )
            self.tokenizer = tokenizer
            
            logger.info(f'skill {model["routing_key"]} loaded to {model_options["device"]}, precision: {model_options["use_precision"]}')
            return { "model_loaded": load_model, "generator": load_model, "tokenizer": tokenizer, "error": load_error }                        
        except Exception as e:
            logger.error(f"error loading model")
            print(e)
            load_error = True
            return { "error": load_error }