from application.base_handler import BaseHandler
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class AsrPipeline(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'audio-url')
        return is_valid, errors
    
    def execute(self, model, request):        
        audio_url = request["audio_url"]
        result = { "text": model["model"](audio_url, max_new_tokens=1024)["text"] }
        logger.info(f"asr extracted text: {result['text']}")
        return result

    def load(self, model, model_options, local_path):
        asr_model = pipeline("automatic-speech-recognition", model=local_path)
        return {"model": asr_model, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}