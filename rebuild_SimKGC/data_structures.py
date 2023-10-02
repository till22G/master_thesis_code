import os
import json

from torch.utils.data import Dataset
from typing import List

from logger import logger

entity_descriptions = {}

def load_entity_descriptions(path) -> None:
    
    assert os.path.exists(path), "Path is invalid"
    assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
    logger.info("Loading entity descriptions from {}".format(path))
    
    with open (path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
            
    for item in data:
        entity_descriptions[item["entity_id"]] = item["entity_desc"]
        
    logger.info("{} entity descriptinos loaded".format(len(entity_descriptions)))
    

def _create_head_text(entity_head: str, entity_desc: str):
    if not entity_desc:
        return entity_head
    if entity_desc.startswith(entity_head):
        entity_desc = entity_desc[len(entity_head):].strip()
    return "{}: {}".format(entity_head, entity_desc)
  
    
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
        
        head, head_desc, tail, tail_desc = self.head, self.head_desc, self.tail, self.tail_desc
        
        head_text = _create_head_text(head, head_desc)
        
    
        return {'hr_token_ids': None,
                'hr_token_type_ids': None,
                'tail_token_ids': None,
                'tail_token_type_ids': None,
                'head_token_ids': None,
                'head_token_type_ids': None,
                'obj': self}
        
    
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
        return self.data_points[index]
            

def load_data(path: str, backward_triples: bool = True) -> List[DataPoint]:
        global entity_descriptions
        if not entity_descriptions:
            load_entity_descriptions("../data/fb15k237/entities.json")
        
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        
        logger.info("Load {} datapoints from {}".format(len(data), path))
        
            
        if backward_triples:
            logger.info("Adding inverse triples")
            
        datapoints = []
        for item in data:
            datapoints.append(DataPoint(item["head_id"],
                                        item["head"],
                                        entity_descriptions[item["head_id"]],
                                        item["relation"], 
                                        item["tail_id"],
                                        entity_descriptions[item["tail_id"]],
                                        item["tail"]))
            if backward_triples:
                datapoints.append(DataPoint(item["tail_id"],
                                            item["tail"],
                                            entity_descriptions[item["tail_id"]],
                                            " ".join(("inverse", item["relation"])), 
                                            item["head_id"],
                                            entity_descriptions[item["head_id"]],
                                            item["head"]))
                
        logger.info("Created dataset with {} datapoints".format(len(datapoints)))
            
        return datapoints
    
def collate_fn(data_batch: List[DataPoint]) -> dict:
    
    hr_token_ids = None
    hr_mask = None
    hr_token_type_id = None
    tail_token_ids = None
    tail_mask = None
    tail_token_type_ids = None
    
    
    
    return {"hr_token_ids" : hr_token_ids,
            "hr_mask" : hr_mask,
            "hr_token_type_id" : hr_token_type_id,
            "tail_token_ids" : tail_token_ids,
            "tail_mask" : tail_mask,
            "tail_token_type_ids" : tail_token_type_ids}