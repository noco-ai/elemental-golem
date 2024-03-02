from transformers.generation.streamers import BaseStreamer
from tqdm import tqdm
import json

class ProgressStreamer(BaseStreamer):
    def __init__(self):
        self.token_count = 0
        self.max_new_tokens = 0
        self.show_bar = False
        self.amqp_config = None     
        self.label = ""

    def put(self, value):
        self.token_count += 1
        if self.show_bar:
            self.progress_bar.update(1)            

        if self.amqp_config:        
            send_body = {
                "total": self.max_new_tokens,
                "current": self.token_count,
                "label": self.label,
                "model": self.model
            }
            
            self.amqp_config["channel"].basic_publish(
                exchange=self.amqp_config["headers"]['return_exchange'], 
                routing_key=self.amqp_config["headers"]['return_routing_key'], 
                body=json.dumps(send_body), properties=self.amqp_config["outgoing_properties"])             

    def end(self):
        if self.show_bar:
            self.progress_bar.close()        

    def configure(self, max_new_tokens, label, model, amqp_config = None, show_bar = True):
        self.max_new_tokens = max_new_tokens
        self.show_bar = show_bar
        self.amqp_config = amqp_config        
        self.token_count = 0
        self.label = label
        self.model = model
        if show_bar:
            self.progress_bar = tqdm(total=max_new_tokens)