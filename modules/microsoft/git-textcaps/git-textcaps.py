from application.base_handler import BaseHandler
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests

class GitTextCaption(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-url')
        return is_valid, errors

    def execute(self, model, request):
        image = Image.open(requests.get(request["img_url"], stream=True).raw)
        pixel_values = model["processor"](images=image, return_tensors="pt").to(model["device"]).pixel_values
        generated_ids = model["model"].generate(pixel_values=pixel_values, max_length=256)
        generated_caption = model["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": generated_caption}

    def load(self, model, model_options, local_path):
        processor = AutoProcessor.from_pretrained(local_path)
        git_model = AutoModelForCausalLM.from_pretrained(local_path)
        return {"model": git_model, "processor": processor, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}