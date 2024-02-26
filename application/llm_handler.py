from urllib.parse import urlparse
from application.base_handler import BaseHandler
import logging

logger = logging.getLogger(__name__)

class LlmHandler(BaseHandler):
    
    def __init__(self):
        super().__init__()

    def load(self, model, model_options, local_path):
        pass
    
    def load_config_settings(self, num_input_tokens, request):
        config = self.model_config
        max_new_tokens_config = int(request.get("max_new_tokens", 1024))
        max_seq_len = config.get("max_seq_len", 2048)
        max_new_tokens = min(max_new_tokens_config, max_seq_len - num_input_tokens)
        top_p = request.get("top_p", 0.9)
        top_k = request.get("top_k", 50)
        seed = request.get("seed", -1)
        min_p = request.get("min_p", 0.05)
        mirostat = request.get("mirostat", 0)
        mirostat_eta = request.get("mirostat_eta", 0.01)
        mirostat_tau = request.get("mirostat_tau", 5)
        temperature = request.get("temperature", 1)
        stream_output = True if "stream" in request and request["stream"] == True else False
        debug = "debug" in request
        stop_key = request.get("stop_key", "<stop>")                

        logger.info(f"prompt tokens: {num_input_tokens}, max completion tokens: {max_new_tokens}, context length: {max_seq_len}")
        logger.info(f"temperature: {temperature}, top_p: {top_p}, top_k: {top_k}, seed: {seed}, stream output: {stream_output}")
        logger.info(f"min_p: {min_p}, mirostat: {mirostat}, mirostat_eta: {mirostat_eta}, mirostat_tau: {mirostat_tau}")
        return max_new_tokens, top_p, top_k, seed, temperature, stream_output, debug, stop_key, min_p, mirostat, mirostat_eta, mirostat_tau    

    def build_stop_conditions(self, stops, to_lower = True):
        check_stop_token = False
        stop_conditions = []
        for stop_text in stops:       
            if stop_text == "<stop>":
                check_stop_token = True
                continue
            add_condition = stop_text.lower() if to_lower == True else stop_text
            stop_conditions.append(add_condition)
        
        return check_stop_token, stop_conditions

    def check_stop_conditions(self, token, res_line, eos_token, check_stop_token, stop_conditions):
        if check_stop_token and token == eos_token:
            return True

        for stop_string in stop_conditions:
            if res_line.lower().endswith(stop_string):
                return True
        
        return False
    
    def finish_response(self, stop_key, response, request, stream_output,
                             finish_reason, tokens_per_second, new_tokens, input_tokens, model, elapsed, debug):
        if debug and stream_output == False:
            print('\033[92m' + response + '\033[0m')

        send_content = ""
        if stream_output:
            send_content = stop_key    
        elif "start_response" in request:
            send_content = f"{request['start_response']}{response}"
        else:
            send_content = response

        llm_response = {"content": send_content, "finish_reason": finish_reason, 
                "tokens_per_second": round(tokens_per_second, 2), "completion_tokens": new_tokens, "prompt_tokens": input_tokens, "model": model }
        
        if debug:
            print(llm_response)

        logger.info(f"prompt processed in {elapsed:.2f} seconds, new tokens: {new_tokens}, tokens/second: {tokens_per_second:.2f}")   
        return llm_response
    
    def get_token_count(self, input_text):
        return 100000
    
    def _get_system_prompt(self, request, config):
        system_prompt = ""
        in_request = False
        contains_user_message = False

        if "system_message" in config and len(config["system_message"]):
            system_prompt = config['system_message']            

        # override with system prompt provided by request
        messages_len = len(request["messages"])
        if messages_len and request["messages"][0]["role"] == "system":
            system_prompt = request['messages'][0]['content']            
            in_request = True

        if "system_prompt_format" in config:            
            template = config["system_prompt_format"]
            ai_role = request["ai_role"] if "ai_role" in request else config["ai_role"]
            user_role = request["user_role"] if "user_role" in request else config["user_role"]                                    
            if "{prompt}" in template:       
                check_index = 1 if in_request else 0
                check_len = 2 if in_request else 1
                prompt = request["messages"][check_index]["content"] if messages_len >= check_len and request["messages"][check_index]["role"] == "user" else ""
                response = request["messages"][check_index + 1]["content"] if check_index + 1 < messages_len and request["messages"][check_index + 1]["role"] == "assistant" else ""
                system_prompt = template.format(user_role=user_role, system_prompt=system_prompt.strip(), ai_role=ai_role, prompt=prompt, response=response) + "\n"
                contains_user_message = True
            else:
                system_prompt = template.format(user_role=user_role, system_prompt=system_prompt.strip(), ai_role=ai_role)
        
        return system_prompt, in_request, contains_user_message
    
    def _prep_prompt(self, request, config):
        request_system_message = None
        max_new_tokens = request.get("max_new_tokens", 1024)
        max_seq_length = config["max_seq_len"]        
        print(max_seq_length)
        print(max_new_tokens)
        max_input_tokens = max(max_seq_length - max_new_tokens, 0)        

        if max_input_tokens == 0:
            logger.error("error with configuration of models context limits")
            raise ValueError('error with configuration of models context limits')
        
        # give a little wiggle room for the way the prompt is built
        max_input_tokens -= 64
        
        system_prompt, sys_prompt_in_request, clip_first_user_message = self._get_system_prompt(request, config)
        system_prompt_tokens = self.get_token_count(system_prompt)
        if system_prompt_tokens >= max_input_tokens:
            logger.error("system prompt excceds max input tokens")
            raise ValueError("system prompt excceds max input tokens")
                
        if sys_prompt_in_request:
            request_system_message = request["messages"][0]
            request["messages"].pop(0) 

        if clip_first_user_message:
            request["messages"].pop(0)

        # clip all but last message if this is an instruct model
        if  len(request["messages"]) == 0:
            messages = []
        if "model_type" in config and config["model_type"] == "instruct":
            messages = [request["messages"][-1]]
        else:
            messages = request["messages"][::-1]      

        return messages, system_prompt_tokens, request_system_message, system_prompt, sys_prompt_in_request, max_input_tokens     

    def build_prompt(self, request, config, model):
        prompt = ""
        
        # raw prompt
        if "raw" in request:
            prompt = request["raw"]
            if "start_response" in request:
                prompt += request["start_response"]            
            return prompt

        messages, system_prompt_tokens, request_system_message, system_prompt, sys_prompt_in_request, max_input_tokens = self._prep_prompt(request, config)        
        max_input_tokens -= 64

        # get delimiter in-between user and prompt and get roles        
        ai_role = request["ai_role"] if "ai_role" in request else config["ai_role"]
        user_role = request["user_role"] if "user_role" in request else config["user_role"]
        template = config["prompt_format"]

        prompt_parts = []  
        input_token_count = system_prompt_tokens

        for index, message in enumerate(messages):
            
            if message["role"] == "assistant":
                continue            
            
            ai_response = "" if index == 0 else messages[index - 1]["content"].strip()
            formatted_string = template.format(user_role=user_role, prompt=message['content'].strip(), ai_role=ai_role, response=ai_response)            
            token_count = self.get_token_count(formatted_string)             
            if input_token_count + token_count > max_input_tokens:
                break

            input_token_count += token_count
            prompt_parts.append(formatted_string)          

        prompt_parts = prompt_parts[::-1]        
        prompt = system_prompt + "\n".join(prompt_parts)        
        if "start_response" in request:
            prompt += request["start_response"]

        return prompt