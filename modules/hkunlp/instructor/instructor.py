from diffusers.utils import load_image
from application.base_handler import BaseHandler
from InstructorEmbedding import INSTRUCTOR

class Instructor(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'instructor')
        return is_valid, errors

    def execute(self, model, request):        
        text = request["text"]
        instruction = request["instruction"]
        embedding = model["model"].encode([[instruction,text]])
        result = {"embedding": embedding.tolist()}
        return result

    def load(self, model, model_options, local_path):
        return {"model": INSTRUCTOR(local_path), "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}