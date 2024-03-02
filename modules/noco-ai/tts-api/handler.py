from application.base_handler import BaseHandler
import logging
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import base64
from io import BytesIO
import requests
import tempfile
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class XTTSHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'voice-gen')
        return is_valid, errors

    def is_valid_url(self, url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def download_temp_file(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            return None
        
    def execute(self, model, request):
        prompt = request.get("prompt", "")
        voice_preset = request.get("voice", "default")
        prompt_length = len(prompt)

        temp_file_path = None
        if self.is_valid_url(voice_preset):
            temp_file_path = self.download_temp_file(voice_preset)
            if temp_file_path:
                voice_preset = temp_file_path
            else:
                voice_preset = model["default_wav"]
        else:
            voice_preset = model["default_wav"]

        logger.info(f"prompt: {prompt}, voice: {voice_preset}, length: {prompt_length}")

        outputs = model["loaded_model"].synthesize(
            prompt,
            model["config"],
            speaker_wav=voice_preset,
            gpt_cond_len=3,
            language="en",
        )
        if temp_file_path:
            os.remove(temp_file_path)
            
        base64_encoded_wav = None
        with BytesIO() as wav_file:
            sf.write(wav_file, outputs["wav"], samplerate=22050, format='WAV')
            wav_file.seek(0)
            binary_wav = wav_file.read()
            base64_encoded_wav = base64.b64encode(binary_wav).decode()

        return {"wav": base64_encoded_wav }
    
    def load(self, model, model_options, local_path):
        self.model_config = model["configuration"]      

        try:                        
            config = XttsConfig()
            config.load_json(f"{local_path}/config.json")
            loaded_model = Xtts.init_from_config(config)
            loaded_model.load_checkpoint(config, checkpoint_dir=local_path, eval=True)
            if model_options["device"] != "cpu":
                loaded_model.cuda(model_options["device"])

            logger.setLevel(logging.INFO)
            return {
                "loaded_model": loaded_model,
                "config": config,
                "default_wav": f"{local_path}/samples/en_sample.wav"
            }
        except Exception as e:
            print(f"error loading xtts model")
            print(e)
            return { "error": True }
