from transformers import Blip2Processor, Blip2ForConditionalGeneration
from application.base_handler import BaseHandler
from PIL import Image
import requests
import torch

class Blip2Opt27b(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'visual-qa')
        return is_valid, errors
    
    def execute(self, model, request):        
        img_url = request["img_url"]        
        image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        prompt = request["text"]

        # build the input tensor
        if self.use_precision == 'half':
            inputs = model["processor"](images=image, text=prompt, return_tensors="pt").to(model["device"], torch.float16)
        else:
            inputs = model["processor"](images=image, text=prompt, return_tensors="pt").to(model["device"])

        generated_ids = model["model"].generate(**inputs, max_new_tokens=256)
        generated_text = model["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return {"text":generated_text}
    
    def load(self, model, model_options, local_path):

        # load the processor        
        processor = Blip2Processor.from_pretrained(local_path)
        self.use_precision = model_options["use_precision"]

        # load the model        
        if self.use_precision == "full":        
            blip2_model = Blip2ForConditionalGeneration.from_pretrained(local_path)            
        elif model_options["use_precision"] == "half":
            blip2_model = Blip2ForConditionalGeneration.from_pretrained(local_path, torch_dtype=torch.float16)

        return {"model": blip2_model, "device": model_options["device"], "processor": processor, "device_memory": model["memory_usage"][self.use_precision]}