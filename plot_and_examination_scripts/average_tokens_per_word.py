import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser(prog="evaluate graph density and token length")
parser.add_argument("--task", default="WN18RR", type=str)

args = parser.parse_args()

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(os.path.dirname(script_dir), "rebuild_SimKGC", "data")

entity_dict = None

def get_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(path=os.path.join(data_dir, args.task, "entities.json"))
    return entity_dict


class EntityDict():
    def __init__(self, path: str, inductive_test_path: str = None) -> None:
        self.entities = []
        self.id2entity = {}
        self.entity2idx = {}
        
        self._load_entity_dict(path, inductive_test_path)
    
    def _load_entity_dict(self, path: str, inductive_test_path: str = None):
        assert os.path.exists(path), "Path is invalid {}:".format(path)
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
        print("Loading entity dict ...")
        with open(path, "r", encoding="utf8") as infile:
            data = json.load(infile)
            
        self.entities = data

        if inductive_test_path:
            with open(inductive_test_path, "r", encoding="utf-8") as infile:
                data = json.load(infile)

            entity_ids = set()
            for triple in data:
                entity_ids.add(triple['head_id'])
                entity_ids.add(triple['tail_id'])
            self.entities = [entity for entity in self.entities if entity["entity_id"] in entity_ids]

        self.id2entity = {entity["entity_id"]: entity for entity in data}
        self.entity2idx = {entity["entity_id"]: i for i, entity in enumerate(self.entities)}

        print("Entity dict successfully loaded")
    
    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]
    
    def get_entity_by_id(self, entity_id: str) -> dict:
        return self.id2entity[entity_id]
    
    def get_entity_by_idx(self, entity_idx: int) -> dict:
        return self.entities[entity_idx]
    
    def __len__(self):
        return len(self.entities)
    


entity_dict = get_entity_dict()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

total_words = 0
total_tokens = 0

for entity in tqdm(entity_dict.entities):
    entity_desc = entity["entity_desc"]
    num_desc_words = len(entity_desc.split())
    total_words += num_desc_words

    tokens = tokenizer(text=entity_desc, add_special_tokens=False,
                       max_length=10000,
                       return_token_type_ids=False,
                       truncation=True)
    
    num_tokens = len(tokens["input_ids"])
    total_tokens += num_tokens

print(total_words)
print(total_tokens)

print("Average number of tokens for a word in {} entity descriptions: {:.4f}".format(args.task, (total_tokens/ total_words)))