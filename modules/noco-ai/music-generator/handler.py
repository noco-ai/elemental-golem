from application.base_handler import BaseHandler
from application.progress_streamer import ProgressStreamer
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import base64
from io import BytesIO
import scipy
import copy
import logging

logger = logging.getLogger(__name__)

class MusicGen(BaseHandler):
    def __init__(self):
        self.progress_streamer = ProgressStreamer()
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'audio-gen')
        return is_valid, errors

    def execute(self, model, request):
        prompt = request.get("prompt", "")  # defaults to an empty string if "prompt" is not in request
        seconds = int(request.get("seconds", 5))
        guidance_scale = int(request.get("guidance_scale", 3))
        send_progress = request.get("progress", True)
        max_new_tokens = seconds * 52

        # prep headers for sending progress data
        if send_progress:
            progress_headers = copy.deepcopy(model["amqp_headers"])
            outgoing_properties = self.copy_queue_headers(progress_headers, "update_progress")
            amqp_config = {
                "headers": progress_headers,
                "outgoing_properties": outgoing_properties,
                "channel": model["amqp_channel"]
            }            
            self.progress_streamer.configure(max_new_tokens, self.model_config["progress_label"], self.routing_key, amqp_config)
        else:
            self.progress_streamer.configure(max_new_tokens, self.model_config["progress_label"], self.routing_key)

        # Assuming the model function can take these parameters:
        logger.info(f"prompt: {prompt}, seconds: {seconds}, max new tokens: {max_new_tokens}, guidance scale: {guidance_scale}")
        inputs = model["processor"](
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(model["device"])        
        audio_values = model["model"].generate(**inputs, do_sample=True, streamer=self.progress_streamer, guidance_scale=guidance_scale, max_new_tokens=max_new_tokens)

        # Save image to an in-memory bytes buffer
        buffered = BytesIO()
        sampling_rate = model["model"].config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(buffered, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
        
        # Convert bytes buffer to a base64-encoded string
        wav_str = base64.b64encode(buffered.getvalue()).decode()
        return {"wav": wav_str}    
    
    def load(self, model, model_options, local_path):        
        self.model_config = model["configuration"]      
        self.routing_key = model["routing_key"]
        processor = AutoProcessor.from_pretrained(local_path)
        load_model = MusicgenForConditionalGeneration.from_pretrained(local_path)        

        return {
            "model": load_model,
            "processor": processor,
            "device": model_options["device"],
            "device_memory": model["memory_usage"][model_options["use_precision"]]
        }
