from application.base_handler import BaseHandler
import torch
import base64
from io import BytesIO
import logging
import torch
from diffusers import StableDiffusionXLPipeline, KDPM2DiscreteScheduler, DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
import copy
import json

logger = logging.getLogger(__name__)

class StableDiffusionXl(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-gen')
        return is_valid, errors

    def get_latents(self, num_images=1, height=1024, width=1024, user_seed=-1, device="cuda:0", model=None):
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

    def step_callback(self, pipeline: DiffusionPipeline, step: int, timestep: int, callback_kwargs):
        if self.stream_progress == False:
            return callback_kwargs    
        
        self.current_step = self.current_step + 1
        label = self.model_config["progress_label"] if "progress_label" in self.model_config else self.routing_key
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

    def execute(self, model, request):
        prompt = request.get("prompt", "")
        height = request.get("height", 1024)
        width = request.get("width", 1024)
        steps = request.get("steps", 50)
        seed = request.get("seed", -1)
        self.stream_progress = request.get("progress", False)
        negative_prompt = request.get("negative_prompt", "")
        guidance_scale = request.get("guidance_scale", 7.5)
        num_images_per_prompt = 1        

        if self.model_config["is_turbo"] == True and steps > 4:
            guidance_scale = 0.0            
            steps = 4
        
        high_noise_frac = 0.8
        if self.stream_progress == True:
            progress_headers = copy.deepcopy(model["amqp_headers"])
            outgoing_properties = self.copy_queue_headers(progress_headers, "update_progress")
            self.amqp_progress_config = {
                "headers": progress_headers,
                "outgoing_properties": outgoing_properties,
                "channel": model["amqp_channel"]
            }
            self.current_step = 0
            if self.model_config["is_turbo"] == False:
                self.total_steps = ((steps * high_noise_frac) * 2) + (steps * (1 - high_noise_frac))
            else:
                self.total_steps = steps

        latent_data = self.get_latents(num_images_per_prompt, height, width, seed, self.model_options["device"], model["model"])
        logger.info(f"prompt: {prompt}, height: {height}, width: {width}, steps: {steps}, guidance scale: {guidance_scale}, seed: {latent_data['seed']}")
        conditioning, pooled = model["compel"](prompt)
        negative_conditioning, negative_pooled = model["compel"](negative_prompt)

        if self.model_config["is_turbo"] == False:            
            conditioning_refiner, pooled_refiner = model["compel_refiner"](prompt)
            negative_conditioning_refiner, negative_pooled_refiner = model["compel_refiner"](negative_prompt)
            base_image = model["model"](prompt_embeds=conditioning, pooled_prompt_embeds=pooled, height=height, width=width, num_inference_steps=steps, callback_on_step_end=self.step_callback, latents=latent_data["latents"], denoising_end=high_noise_frac,
                            negative_prompt_embeds=negative_conditioning, negative_pooled_prompt_embeds=negative_pooled, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt, output_type="latent").images        
            image = model["refiner"](prompt_embeds=conditioning_refiner, pooled_prompt_embeds=pooled_refiner, negative_prompt_embeds=negative_conditioning_refiner,
                            negative_pooled_prompt_embeds=negative_pooled_refiner, num_inference_steps=steps, denoising_start=high_noise_frac, image=base_image, callback_on_step_end=self.step_callback).images[0]
        else:
            image = model["model"](prompt_embeds=conditioning, pooled_prompt_embeds=pooled, height=height, width=width, num_inference_steps=steps, latents=latent_data["latents"], 
                            guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt, callback_on_step_end=self.step_callback).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG") 

        # Convert bytes buffer to a base64-encoded string
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"image": img_str, "seed": latent_data["seed"], "guidance_scale": guidance_scale, "steps": steps }
    
    def load(self, model, model_options, local_path):
        self.model_options = model_options        
        self.model_config = model["configuration"]
        self.routing_key = model["routing_key"]

        is_turbo = model["configuration"]["is_turbo"]
        if "civitai" not in local_path:             
            logger.info("loading sd xl model")           
            load_model = StableDiffusionXLPipeline.from_pretrained(local_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")            
        else:            
            logger.info("loading civit sd xl model")
            load_model = StableDiffusionXLPipeline.from_single_file(local_path, torch_dtype=torch.float16, variant="fp16")        

        compel = Compel(
            tokenizer=[load_model.tokenizer, load_model.tokenizer_2] ,
            text_encoder=[load_model.text_encoder, load_model.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

        ret = {
            "model": load_model,
            "device": model_options["device"],            
            "device_memory": model["memory_usage"][model_options["use_precision"]],
            "compel": compel
        }

        # load the refiner model
        if is_turbo == False:            
            load_model.scheduler = KDPM2DiscreteScheduler.from_config(load_model.scheduler.config)
            logger.info("loading sd xl refiner")           
            load_refiner = DiffusionPipeline.from_pretrained(
                "./data/models/stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=load_model.text_encoder_2,
                vae=load_model.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            load_refiner.to(model_options["device"])
            ret["refiner"] = load_refiner

            compel_refiner = Compel(
                tokenizer=[load_refiner.tokenizer_2],
                text_encoder=[load_refiner.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[True],
            )
            ret["compel_refiner"] = compel_refiner

        return ret
