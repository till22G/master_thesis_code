import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def build_model(args) -> nn.Module:
    return CustomModel(args)

class CustomModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.args = args
        
        self.bert_hr = AutoModel.from_pretrained(args.pretrained_model) # create bert model for relation aware embeddings
        self.bert_t = AutoModel.from_pretrained(args.pretrained_model) # create bert model for tail entity embeddings
        
    def _encode(self, model, input_token_ids, input_mask, input_token_type_ids):
        output = model(input_ids=input_token_ids,
                       attention_mask=input_mask,
                       token_type_ids=input_token_type_ids,
                       return_dict=True)
        last_hidden_state = output.last_hidden_state
        print(last_hidden_state.size())
        cls_output = last_hidden_state[:, 0, :]
        print(cls_output)
        
        return ""
        
    def forward(self, batched_hr_token_ids, batched_hr_mask, batched_hr_token_type_ids,
                batched_tail_token_ids, batched_tail_mask, batched_tail_token_type_ids, **kwargs): 

        hr_vec = self._encode(self.bert_hr,
                              batched_hr_token_ids,
                              batched_hr_mask,
                              batched_hr_token_type_ids)
        
        t_vec = self._encode(self.bert_t,
                             batched_tail_token_ids,
                             batched_tail_mask,
                             batched_tail_token_type_ids)
        
        return {"hr_vec" : hr_vec,
                "t_vec" : t_vec}
        