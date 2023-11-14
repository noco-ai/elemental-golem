from diffusers.utils import load_image
from application.base_handler import BaseHandler
from transformers import pipeline

class ImageToTextPipeline(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-url')
        return is_valid, errors
    
    def execute(self, model, request):
        predict = model["model"](request["img_url"])        
        return predict

    def load(self, model, model_options, local_path):
        vit_model = pipeline("image-to-text", model=local_path)
        return {"model": vit_model, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}