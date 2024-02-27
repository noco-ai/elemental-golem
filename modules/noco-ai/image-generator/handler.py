from application.base_handler import BaseHandler
from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler
import torch
import base64
from io import BytesIO
import logging
import json
import copy
from compel import Compel

logger = logging.getLogger(__name__)

class StableDiffusion(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-gen')
        return is_valid, errors

    def step_callback(self, pipeline: StableDiffusionPipeline, step: int, timestep: int, callback_kwargs):
        if self.stream_progress == False:
            return callback_kwargs    
        
        self.current_step = self.current_step + 1
        label = self.model_configuration["progress_label"] if "progress_label" in self.model_configuration else self.routing_key
        send_body = {
            "total": self.total_steps,
            "current": self.current_step,
            "label": label,
            "model": self.routing_key
        }
            
        self.amqp_progress_config["channel"].basic_publish(
            exchange=self.amqp_progress_config["headers"]['return_exchange'], 
            routing_key=self.amqp_progress_config["headers"]['return_routing_key'], 
            body=json.dumps(send_body), properties=self.amqp_progress_config["outgoing_properties"])             
        
        return callback_kwargs
    
    def get_latents(self, num_images=1, height=512, width=512, user_seed=-1, device="cuda:0", model=None):
        latents = None
        generator = torch.Generator(device=device)        
        if user_seed == -1:
            seed = generator.seed()
        else:
            seed = user_seed
        generator = generator.manual_seed(seed)
        
        latents = torch.randn(
            (num_images, model.unet.in_channels, height // 8, width // 8),
            generator = generator,
            device = device,
            dtype = torch.float16
        )
        return { "seed": seed, "latents": latents }

    def execute(self, model, request):
        prompt = request.get("prompt", "")
        height = request.get("height", 512)
        width = request.get("width", 512)
        steps = request.get("steps", 50)
        seed = request.get("seed", -1)
        self.stream_progress = request.get("progress", False)
        negative_prompt = request.get("negative_prompt", "")
        guidance_scale = request.get("guidance_scale", 7.5)
        num_images_per_prompt = 1        

        if self.stream_progress == True:
            progress_headers = copy.deepcopy(model["amqp_headers"])
            outgoing_properties = self.copy_queue_headers(progress_headers, "update_progress")
            self.amqp_progress_config = {
                "headers": progress_headers,
                "outgoing_properties": outgoing_properties,
                "channel": model["amqp_channel"]
            }
            self.current_step = 0            
            self.total_steps = steps * 2

        latent_data = self.get_latents(num_images_per_prompt, height, width, seed, self.model_options["device"], model["model"])
        logger.info(f"prompt: {prompt}, height: {height}, width: {width}, steps: {steps}, guidance scale: {guidance_scale}, seed: {latent_data['seed']}")

        prompt_embeds = model["compel"](prompt)
        negative_prompt_embeds = model["compel"](negative_prompt)
        image = model["model"](prompt_embeds=prompt_embeds, height=height, width=width, num_inference_steps=steps, latents=latent_data["latents"], callback_on_step_end=self.step_callback,
                            negative_prompt_embeds=negative_prompt_embeds, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG") 

        # Convert bytes buffer to a base64-encoded string
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"image": img_str, "seed": latent_data["seed"], "guidance_scale": guidance_scale, "steps": steps }
    
    def load(self, model, model_options, local_path):
        self.model_options = model_options
        self.routing_key = model["routing_key"]
        self.model_configuration = model["configuration"]

        try:
            if "civitai" not in local_path: 
                logger.info("loading standard sd model")           
                load_model = StableDiffusionPipeline.from_pretrained(local_path, torch_dtype=torch.float16, safety_checker=None)
            else:
                logger.info("loading civit sd model")
                load_model = StableDiffusionPipeline.from_single_file(local_path, load_safety_checker=False, torch_dtype=torch.float16)
            
            load_model.scheduler = KDPM2DiscreteScheduler.from_config(load_model.scheduler.config)
            compel = Compel(tokenizer=load_model.tokenizer, text_encoder=load_model.text_encoder)

            return {
                "model": load_model,
                "device": model_options["device"],
                "device_memory": model["memory_usage"][model_options["use_precision"]],
                "compel": compel
            }
    
        except Exception as e:
            print(f"error loading sd model")
            print(e)
            return { "error": True }
