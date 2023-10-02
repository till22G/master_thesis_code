import os
import json

from torch.utils.data import Dataset
from typing import List

from logger import logger

class DataPoint():
    def __init__(self, 
                 head_id: str = None, 
                 head_desc: str = None, 
                 relation: str = None, 
                 tail_id: str = None, 
                 tail_desc: str = None) -> None:

        self.head_id = head_id
        self.head_desc = head_desc
        self.relation = relation
        self.tail_id = tail_id
        self.tail_desc = tail_desc
    
    def get_head_id(self) -> str:
        if self.head_id is not None:
            return self.head_id
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
        
    def get_tail_desc(self) -> str:
        if self.tail_desc is not None:
            return self.tail_desc
        else:
            return ""
        
    def encode_to_dict(self) -> dict:
        
    
    
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
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        
        logger.info("Load {} datapoints from {}".format(len(data), path))
        if backward_triples:
            logger.info("Adding inverse triples")
            
        datapoints = []
        for item in data:
            datapoints.append(DataPoint(item["head_id"],
                                        item["head"],
                                        item["relation"], 
                                        item["tail_id"],
                                        item["tail"]))
            if backward_triples:
                datapoints.append(DataPoint(item["tail_id"],
                                            item["tail"],
                                            " ".join(("inverse", item["relation"])), 
                                            item["head_id"],
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