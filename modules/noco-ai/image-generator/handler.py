from application.base_handler import BaseHandler
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class StableDiffusion(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-gen')
        return is_valid, errors

    def execute(self, model, request):
        prompt = request.get("prompt", "")  # defaults to an empty string if "prompt" is not in request
        height = request.get("height", 512)
        width = request.get("width", 512)
        steps = request.get("steps", 50)
        negative_prompt = request.get("negative_prompt", "")
        guidance_scale = request.get("guidance_scale", 7.5)
        num_images_per_prompt = 1        

        # Assuming the model function can take these parameters:
        logger.info(f"prompt: {prompt}, height: {height}, width: {width}, steps: {steps}, guidance scale: {guidance_scale}")
        image = model["model"](prompt, height=height, width=width, num_inference_steps=steps, 
                            negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images[0]

        # Save image to an in-memory bytes buffer
        buffered = BytesIO()
        image.save(buffered, format="PNG")  # Assuming it's a PNG format, change if needed

        # Convert bytes buffer to a base64-encoded string
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"image": img_str}
    
    def load(self, model, model_options, local_path):
        if "civitai" not in local_path: 
            logger.info("loading standard sd model")           
            load_model = StableDiffusionPipeline.from_pretrained(local_path, torch_dtype=torch.float16, safety_checker=None)
        else:
            logger.info("loading civit sd model")
            load_model = StableDiffusionPipeline.from_single_file(local_path, load_safety_checker=False, torch_dtype=torch.float16)
        
        return {
            "model": load_model,
            "device": model_options["device"],
            "device_memory": model["memory_usage"][model_options["use_precision"]]
        }
