from diffusers.utils import load_image
from application.base_handler import BaseHandler
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import torch

class FacebookConvNet(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-url')
        return is_valid, errors
    
    def execute(self, model, request):        
        raw_image = load_image(request["img_url"]).convert('RGB')
        inputs = model["feature_extractor"](raw_image, return_tensors="pt")

        with torch.no_grad():
            inputs = {key: value.to(model["device"]) for key, value in inputs.items()}
            logits = model["model"](**inputs).logits

        predicted_label = logits.argmax(-1).item()        
        return {"classes": [{"label": model["model"].config.id2label[predicted_label], "score": 1}]}

    def load(self, model, model_options, local_path):
        feature_extractor = ConvNextImageProcessor.from_pretrained(local_path)
        conv_model = ConvNextForImageClassification.from_pretrained(local_path)
        return {"model": conv_model, "feature_extractor": feature_extractor, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}