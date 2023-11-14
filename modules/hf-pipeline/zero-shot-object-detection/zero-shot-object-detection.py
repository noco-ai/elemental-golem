from diffusers.utils import load_image
from application.base_handler import BaseHandler
from transformers import pipeline

class ZeroShotObjectDetectionPipeline(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'zero-shot-img')
        return is_valid, errors
    
    def execute(self, model, request):
        labels = request["labels"]
        predict = model["model"](request["img_url"], candidate_labels = labels)        
        return predict

    def load(self, model, model_options, local_path):
        object_model = pipeline("zero-shot-object-detection", model=local_path)
        return {"model": object_model, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}