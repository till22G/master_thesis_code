import os
import json
import torch

from torch.utils.data import Dataset
from typing import List, Optional
from transformers import AutoTokenizer

from logger import logger
from argparser import args


tokenizer: AutoTokenizer = None

entities = {}
training_triples = []
neigborhood_graph = None


class NeighborhoodGraph():
    def __init__(self) -> None:
        self.graph = {}
        # !!!!!!!!!!!!!! remember to change path !!!!!!!!!!!!!!!!!!!!!!1
        path = "../data/FB15k237/train.json"
        # !!!!!!!!!!!!!! remember to change path !!!!!!!!!!!!!!!!!!!!!!1
        logger.info("Building neighborhood graph from {}".format(path))
        if not training_triples:
            load_training_triples(path)
            
        for item in training_triples:
            if item["head_id"] not in self.graph:
                self.graph[item["head_id"]] = set()
            self.graph[item["head_id"]].add(item["tail_id"])
           
            if item["tail_id"] not in self.graph:
                self.graph[item["tail_id"]] = set()
            self.graph[item["tail_id"]].add(item["head_id"])
        
        logger.info("Neighborhood graph succesfully build")
            
        
    def get_neighbors(self, entity_id: str, num_neigbhours: int = 10):
        neigbours = sorted(self.graph.get(entity_id, set()))
        return neigbours[:num_neigbhours]
    

def build_neighborhood_graph():
    global neigborhood_graph
    neigborhood_graph =  NeighborhoodGraph()
    
    
def load_training_triples(path) -> None:
    
    assert os.path.exists(path), "Path is invalid"
    assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
    logger.info("Loading training triple to add neigbours from {}".format(path))
    
    with open(path, "r", encoding="utf-8") as inflile:
        data = json.load(inflile)
    
    for item in data:
        training_triples.append(item)
    
    
def load_entities(path) -> None:
    
    assert os.path.exists(path), "Path is invalid"
    assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"

    logger.info("Loading entity descriptions from {}".format(path))
    
    with open (path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
            
    for item in data:
        entities[item["entity_id"]] = item
        
    logger.info("{} entity descriptinos loaded".format(len(entities)))
    

def _concat_name_desciption(entity_head: str, entity_desc: str):
    if not entity_desc:
        return entity_head
    if entity_desc.startswith(entity_head):
        entity_desc = entity_desc[len(entity_head):].strip()
    return "{}: {}".format(entity_head, entity_desc)


def _tokenize_text(text:str, relation: Optional[str] = None) -> dict:
    global tokenizer
    if tokenizer is None:
        create_tokenizer()
    
    tokens = tokenizer(text,
                       text_pair = relation,
                       add_special_tokens = True,
                       truncation=True,
                       max_length = args.max_number_tokens,
                       return_token_type_ids = True)
    
    return tokens


def create_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    logger.info("Created tokenizer from {}".format(args.pretrained_model))

    
class DataPoint():
    def __init__(self, 
                 head_id: str = None, 
                 head: str = None,
                 head_desc: str = None,
                 relation: str = None, 
                 tail_id: str = None, 
                 tail: str = None,
                 tail_desc: str = None) -> None: 

        self.head_id = head_id
        self.head = head
        self.head_desc = head_desc
        self.relation = relation
        self.tail_id = tail_id
        self.tail = tail
        self.tail_desc = tail_desc
    
    def get_head_id(self) -> str:
        if self.head_id is not None:
            return self.head_id
        else:
            return ""
    
    def get_head(self) -> str:
        if self.head is not None:
            return self.head
        else:
            return ""
        
    def get_head_desc(self) -> str:
        if self.head_desc is not None:
            return self.head_desc
        else:
            return ""
        
    def get_relation(self) -> str:
        if self.relation is not None:
            return self.relation
        else:
            return ""
    
    def get_tail_id(self) -> str:
        if self.tail_id is not None:
            return self.tail_id
        else:
            return ""
        
    def get_tail(self) -> str:
        if self.tail is not None:
            return self.tail
        else:
            return ""
        
    def get_tail_desc(self) -> str:
        if self.tail is not None:
            return self.tail_desc
        else:
            return ""
        
    def encode_to_dict(self) -> dict:
        
        head_desc, tail_desc = self.head_desc, self.tail_desc
        
        # still need to code what happens when this is set to true
        if args.use_neighbors:
            # the entity names should be padded with information from neighbour if
            # their own description is to short
            if len(head_desc.split()) < 20:
                head_desc = " ".join((head_desc, add_neighbor_names(self.head_id, self.tail_id)))
            if len(tail_desc.split()) < 20:
                tail_desc = " ".join((tail_desc, add_neighbor_names(self.tail_id, self.head_id)))
                  
        head_text = _concat_name_desciption(self.head, head_desc)
        hr_tokens = _tokenize_text(head_text, self.relation)
        t_tokens = _tokenize_text(self.tail)
        h_tokens = _tokenize_text(self.head)
                 
        return {'hr_token_ids': hr_tokens["input_ids"],
                'hr_token_type_ids': hr_tokens["token_type_ids"],
                'tail_token_ids': t_tokens["input_ids"],
                'tail_token_type_ids': t_tokens["token_type_ids"],
                'head_token_ids': h_tokens["input_ids"],
                'head_token_type_ids': h_tokens["token_type_ids"],
                'obj': self}
        

# I still need to check this, but I seems that they only build a neighborhood graph 
# from the training set and also use that for adding neighbors during validation
def add_neighbor_names(head_id, tail_id):
    global neigborhood_graph
    if neigborhood_graph is None:
        build_neighborhood_graph()
    neighbor_ids = neigborhood_graph.get_neighbors(head_id)
    
    if tail_id in neighbor_ids:
        neighbor_ids.remove(tail_id)
    
    neighbor_names = [entities[entity_id].get("entity", "") for entity_id in neighbor_ids]
    return " ".join(neighbor_names)


class Dataset(Dataset):
    def __init__(self, path, data_points=None) -> None:
        super().__init__()
        
        self.path = path 
        assert os.path.exists(self.path), "Path is invalid"
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
        
        if data_points is None:
            self.data_points = []
            self.data_points = load_data(self.path)
            
        else:
            data_points = data_points
            
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, index) -> DataPoint:
        return self.data_points[index].encode_to_dict()
            

def load_data(path: str, backward_triples: bool = True) -> List[DataPoint]:
        global entities
        if not entities:
            load_entities("../data/fb15k237/entities.json")
        
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        
        logger.info("Load {} datapoints from {}".format(len(data), path))
        
            
        if backward_triples:
            logger.info("Adding inverse triples")
            
        datapoints = []
        for item in data:
            datapoints.append(DataPoint(item["head_id"],
                                        item["head"],
                                        entities[item["head_id"]].get("entity_desc", ""),
                                        item["relation"], 
                                        item["tail_id"],
                                        entities[item["tail_id"]].get("entity_desc", ""),
                                        item["tail"]))
            if backward_triples:
                datapoints.append(DataPoint(item["tail_id"],
                                            item["tail"],
                                            entities[item["tail_id"]].get("entity_desc", ""),
                                            " ".join(("inverse", item["relation"])), 
                                            item["head_id"],
                                            entities[item["head_id"]].get("entity_desc", ""),
                                            item["head"]))
                
        logger.info("Created dataset with {} datapoints".format(len(datapoints)))
            
        return datapoints
    
def collate_fn(batch: List[DataPoint]) -> dict:
    
    if tokenizer is None:
        create_tokenizer()
    
    hr_token_ids, hr_mask = batch_token_ids_and_mask(
        [torch.LongTensor(datapoint["hr_token_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id)
    
    tail_token_ids, tail_mask = batch_token_ids_and_mask(
        [torch.LongTensor(datapoint["tail_token_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id)
    
    hr_token_type_ids = batch_token_ids_and_mask (
        [torch.LongTensor(datapoint["hr_token_type_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id, create_mask = False)
    
    tail_token_type_ids = batch_token_ids_and_mask (
        [torch.LongTensor(datapoint["tail_token_type_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id, create_mask = False)

    return {"batched_hr_token_ids" : hr_token_ids,
            "batched_hr_mask" : hr_mask,
            "batched_hr_token_type_ids" : hr_token_type_ids,
            "batched_tail_token_ids" : tail_token_ids,
            "batched_tail_mask" : tail_mask,
            "batched_tail_token_type_ids" : tail_token_type_ids}
    

def batch_token_ids_and_mask(data_batch_tensor, pad_token_id=0, create_mask=True):
    max_length = max([item.size(0) for item in data_batch_tensor])
    num_samples =len(data_batch_tensor)
    batch = torch.LongTensor(num_samples, max_length).fill_(0)
    if create_mask:
        mask = torch.ByteTensor(num_samples, max_length).fill_(0)
    for i, tensor in enumerate(data_batch_tensor):
        batch[i, :len(tensor)] = tensor
        if create_mask:
            mask[i, :len(tensor)].fill_(1)
    
    if create_mask:
        return batch, mask
    else:
        return batch
    
