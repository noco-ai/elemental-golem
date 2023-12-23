from application.base_handler import BaseHandler
from transformers import AutoProcessor, AutoModel
import torch
import base64
from io import BytesIO
import scipy
import copy
from application.progress_streamer import ProgressStreamer
import logging

logger = logging.getLogger(__name__)

class BarkHandler(BaseHandler):
    def __init__(self):
        self.progress_streamer = ProgressStreamer()
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'voice-gen')
        return is_valid, errors

    def execute(self, model, request):
        prompt = request.get("prompt", "")          
        send_progress = request.get("progress", True)
        voice_preset = request.get("voice", "v2/en_speaker_1")
        prompt_length = len(prompt)

        if send_progress:
            progress_headers = copy.deepcopy(model["amqp_headers"])
            outgoing_properties = self.copy_queue_headers(progress_headers, "update_progress")
            amqp_config = {
                "headers": progress_headers,
                "outgoing_properties": outgoing_properties,
                "channel": model["amqp_channel"]
            }            
            self.progress_streamer.configure(prompt_length * 25, self.model_config["progress_label"], amqp_config, False)
        else:
            self.progress_streamer.configure(prompt_length * 25, self.model_config["progress_label"], None, False)

        # Assuming the model function can take these parameters:        
        logger.info(f"prompt: {prompt}, voice: {voice_preset}, length: {prompt_length}")        
        inputs = model["processor"](
            text=[prompt],
            voice_preset=voice_preset,
            return_tensors="pt",
        ).to(model["device"])
        speech_values = model["model"].generate(**inputs, do_sample=True, streamer=self.progress_streamer)

        # Save image to an in-memory bytes buffer
        buffered = BytesIO()
        sampling_rate = model["model"].generation_config.sample_rate
        scipy.io.wavfile.write(buffered, rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())
        
        # Convert bytes buffer to a base64-encoded string
        wav_str = base64.b64encode(buffered.getvalue()).decode()
        return {"wav": wav_str}
    
    def load(self, model, model_options, local_path):
        self.model_config = model["configuration"]      
        processor = AutoProcessor.from_pretrained(local_path)
        load_model = AutoModel.from_pretrained(local_path)
        
        return {
            "model": load_model,
            "processor": processor,
            "device": model_options["device"],
            "device_memory": model["memory_usage"][model_options["use_precision"]]
        }
