import torch.nn as nn
import torch

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig, BertConfig, BertModel
from logger import logger
from help_functions import move_to_cuda
from data_structures import construct_triplet_mask

def build_model(args) -> nn.Module:
    logger.info("Building model")
    return CustomModel(args)

def build_model(args) -> nn.Module:
    logger.info("Building model ...")
    return CustomModel(args)

class CustomModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.args = args
        
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        self.pre_batch_weight = args.pre_batch_weight
        self.use_self_negatives = args.use_self_negatives
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        
        self.offset = 0
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        self.pre_batch_datapoints = [None] * num_pre_batch_vectors
        
        # register a buffer with the size of all pre=batch logits held (batch-size * pre-batches)
        # times the size of the last hidden layer (size of embedding)
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        
        if args.custom_model_init:
            config = BertConfig()
            self.bert_hr = BertModel(config)
            self.bert_t = BertModel(config)

        else:
            self.bert_hr = AutoModel.from_pretrained(args.pretrained_model) # create bert model for relation aware embeddings
            self.bert_t = AutoModel.from_pretrained(args.pretrained_model) # create bert model for tail entity embeddings
        
    def _encode(self, model, input_token_ids, input_mask, input_token_type_ids):

        if self.args.pretrained_model == "distilbert-base-uncased":
            output = model(input_ids=input_token_ids,
                              attention_mask=input_mask,
                              return_dict=True)

        else:
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
        output = nn.functional.normalize(output, dim=1) # normlaize result
        
        return output
        
    def forward(self, batched_hr_token_ids, batched_hr_mask, batched_hr_token_type_ids,
                      batched_tail_token_ids, batched_tail_mask, batched_tail_token_type_ids,
                      batched_head_token_ids, batched_head_mask, batched_head_token_type_ids,
                      encode_tail_only = False, encode_hr_only = False,
                      only_ent_embedding = False, **kwargs):        

        if only_ent_embedding:
            encode_tail_only = True

        if not encode_hr_only:
            t_vec = self._encode(self.bert_t,
                                batched_tail_token_ids,
                                batched_tail_mask,
                                batched_tail_token_type_ids
                                )
        
        if encode_tail_only:
            return {'ent_vectors': t_vec.detach()}
            
        hr_vec = self._encode(self.bert_hr,
                              batched_hr_token_ids,
                              batched_hr_mask,
                              batched_hr_token_type_ids
                              )
        
        if encode_hr_only:
            return {'hr_vec': hr_vec.detach()}
        
        h_vec = self._encode(self.bert_t,
                             batched_head_token_ids,
                             batched_head_mask,
                             batched_head_token_type_ids
                             )
        
        return {"hr_vector" : hr_vec,
                "tail_vector" : t_vec,
                "head_vector" : h_vec}
    
    
    def compute_logits(self, encodings: dict, batch_data: dict) -> dict:
        #encodings = output_dict
        #batch_data = batch_dict
        hr_vec, t_vec = encodings["hr_vector"], encodings["tail_vector"]
        labels = torch.arange(hr_vec.size(0)).to(hr_vec.device)

        logits = hr_vec.mm(t_vec.t()) # calculate cos-similarity
        if self.training:
            logits -= torch.zeros(logits.shape).fill_diagonal_(self.add_margin).to(logits.device) # subtract margin
        logits *= self.log_inv_t.exp() # scale with temeratur parameter
    
        triplet_mask = batch_data.get('triplet_mask', None)
        if torch.cuda.is_available(): triplet_mask = move_to_cuda(triplet_mask)
        
        if triplet_mask is not None:
            logits.masked_fill_(triplet_mask, -1e4)

        # add pre batch logits here
        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vec, t_vec, batch_data)
            logits = torch.cat([logits, pre_batch_logits], dim=1)
                        
        # add self negatives here
        if self.use_self_negatives and self.training:
            hr_vec, head_vec = encodings["hr_vector"], encodings["head_vector"]
            self_neg_logits = torch.sum(hr_vec * head_vec, dim=1) * self.log_inv_t.exp()
            self_neg_mask = batch_data["self_neg_mask"]
            if torch.cuda.is_available(): self_neg_mask = move_to_cuda(self_neg_mask)
            self_neg_logits = self_neg_logits.masked_fill_(self_neg_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=1)
            
        return {"logits" : logits,
                "labels" : labels,
                "inv_t" : self.log_inv_t.detach().exp(),
                "hr_vector" : hr_vec.detach(),
                "tail_vector" : t_vec.detach()}

    def _compute_pre_batch_logits(self, hr_vec, t_vec, batch_data):
        batched_datapoints = [datapoint for datapoint in batch_data["batched_datapoints"]]
        pre_batch_logits = hr_vec.mm(self.pre_batch_vectors.clone().t()) # calculate logits with pre-batch vectors and current encodings
        pre_batch_logits *= self.pre_batch_weight * self.log_inv_t.exp() # apply pre-batch weight and scale with temperature

        # mask pre-batch logits
        if self.pre_batch_datapoints[-1] is not None:
            pre_batch_triplet_mask = construct_triplet_mask(batched_datapoints, self.pre_batch_datapoints).to(hr_vec.device)
            pre_batch_logits.masked_fill_(pre_batch_triplet_mask, -1e4)

        # update pre-batched datapoints with new w
        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = t_vec.data.clone()
        self.pre_batch_datapoints[self.offset:(self.offset + self.batch_size)] = batched_datapoints
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_datapoints)

        return pre_batch_logits