from application.base_handler import BaseHandler
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

class E5V2(BaseHandler):
    def __init__(self):
        super().__init__()

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def validate(self, request):
        return True, []

    def execute(self, model, request):        
        text = request["text"]
        if type(text) is str:
            text = [text]

        ret_embed = {}
        for embed in text: 
            batch_dict = model["tokenizer"](f"query: {embed}", max_length=512, padding=True, truncation=True, return_tensors='pt')
            for key in batch_dict:
                batch_dict[key] = batch_dict[key].to(model["device"])

            outputs = model["model"](**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            ret_embed[embed] = embeddings.tolist() 
            
        return {"embeddings": ret_embed}

    def load(self, model, model_options, local_path):
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        e5_model = AutoModel.from_pretrained(local_path)
        return {"model": e5_model, "tokenizer": tokenizer, "device": model_options["device"], "device_memory": model["memory_usage"][model_options["use_precision"]]}