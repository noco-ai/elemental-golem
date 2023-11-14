from application.base_handler import BaseHandler
from transformers import pipeline

class  ImageClassificationPipeline(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-url')
        return is_valid, errors
    
    def execute(self, model, request):        
        img_url = request["img_url"]        
        result = { "classes": model["model"](img_url) }        
        return result

    def load(self, model, model_options, local_path):
        img_model = pipeline("image-classification", model=local_path)
        return {"model": img_model, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}