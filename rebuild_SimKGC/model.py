import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def build_model(args) -> nn.Module:
    return CustomModel(args)

class CustomModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.args = args
        
        model_type = "bert-base-uncased"
        tokenizer  = AutoTokenizer.from_pretrained(model_type)
        bert_hr = AutoModel.from_pretrained(model_type) # create bert model for relation aware embeddings
        bert_t = AutoModel.from_pretrained(model_type) # create bert model for tail entity embeddings
    
    def forward():
        pass 
    