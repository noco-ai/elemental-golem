from application.base_handler import BaseHandler
from transformers import pipeline

class VisualQaPipeline(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'visual-qa')
        return is_valid, errors
    
    def execute(self, model, request):        
        text = request["text"]
        predict = model["model"](question=text, image=request["img_url"])
        return predict

    def load(self, model, model_options, local_path):
        vit_model = pipeline("visual-question-answering", model=local_path)
        return {"model": vit_model, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}