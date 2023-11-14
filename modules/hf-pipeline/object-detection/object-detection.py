import requests
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from diffusers.utils import load_image
from application.base_handler import BaseHandler
from transformers import pipeline

class ObjectDetectionPipeline(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        is_valid, errors = self.validate_request(request, 'img-url')
        return is_valid, errors
    
    def image_with_boxes(self, img_url, detections):
        # Fetch the image
        response = requests.get(img_url)
        img = Image.open(io.BytesIO(response.content))
        
        # Prepare for drawing on the image
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        #font = ImageFont.truetype("arial.ttf", 15)

        # Draw boxes and labels
        for detection in detections:
            label = detection['label']
            box = detection['box']
            score = detection['score']
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red")
            
            # Draw the label with its score
            text = f"{label} {score:.2f}"
            draw.text((xmin, ymin - 20), text, font=font, fill="red")

        # Convert the modified image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return img_base64
        
    def execute(self, model, request):
        detections = model["model"](request["img_url"])
        img_base64 = self.image_with_boxes(request["img_url"], detections)
        return { "objects": detections, "image": img_base64 }                        

    def load(self, model, model_options, local_path):
        detr_model = pipeline("object-detection", model=local_path)
        return {"model": detr_model, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}