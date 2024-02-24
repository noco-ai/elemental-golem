from application.base_handler import BaseHandler
from pika import BasicProperties
from openai import OpenAI
import logging
import time

logger = logging.getLogger(__name__)

class OpenAIImageGeneration(BaseHandler):
    def __init__(self):        
        super().__init__()

    def validate(self, request):        
        is_valid, errors = self.validate_request(request, 'img-gen')
        return is_valid, errors    
    
    def update_config(self, config_data):
        current_config = self.model_config
        merged_config = {**current_config, **config_data}
        self.model_config = merged_config
    
    def execute(self, model, request):        
        prompt = request.get("prompt", "")
        height = request.get("height", 1024)
        width = request.get("width", 1024)

        if height == 512 or width == 512:
            size = "512x512"
        else:
            size = "1024x1024"

        print("generating image")
        response = model["client"].images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        return {"image": response.data[0].b64_json, "seed": 0, "guidance_scale": 0, "steps": 1 }

    def load(self, model, model_options, local_path):         
        self.model_config = model["configuration"]            
        client = OpenAI(
            api_key=model["secrets"]["token"]
        )
        return { "model_name": model["configuration"]["model"], "client": client }
