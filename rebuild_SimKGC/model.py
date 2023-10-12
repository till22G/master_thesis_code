import torch.nn as nn
import torch

from transformers import AutoModel, AutoTokenizer

from triplet_mask import construct_triplet_mask

def build_model(args) -> nn.Module:
    return CustomModel(args)

class CustomModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.args = args
        
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        
        self.bert_hr = AutoModel.from_pretrained(args.pretrained_model) # create bert model for relation aware embeddings
        self.bert_t = AutoModel.from_pretrained(args.pretrained_model) # create bert model for tail entity embeddings
        
    def _encode(self, model, input_token_ids, input_mask, input_token_type_ids):
        output = model(input_ids=input_token_ids,
                       attention_mask=input_mask,
                       token_type_ids=input_token_type_ids,
                       return_dict=True)
        
        last_hidden_states = output.last_hidden_state
        
        # perform mean pooling 
        mask_exp = input_mask.unsqueeze(-1).expand(last_hidden_states.size()).long()
        sum_masked_output = torch.sum(last_hidden_states * mask_exp, 1)
        sum_mask = torch.clamp(mask_exp.sum(1), min=1e-4)
        output = sum_masked_output / sum_mask     
        output = nn.functional.normalize(output, 1) # normlaize result
        
        return output
        
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
        
        
    def compute_logits(self, encodings: dict, batch_data: dict) -> dict: 
        hr_vec, t_vec = encodings["hr_vec"], encodings["t_vec"]
        
        logits = hr_vec.mm(t_vec.t()) # calculate cos-similarity
        if self.training:
            logits -= torch.zeros(logits.shape).fill_diagonal_(self.add_margin).to(logits.device) # subtract margin
        logits *= self.log_inv_t.exp() # scale with temeratur parameter
    

        batched_datapoints = [datapoint["obj"] for datapoint in batch_data["batched_datapoints"]]
        triplet_mask = construct_triplet_mask(batched_datapoints)