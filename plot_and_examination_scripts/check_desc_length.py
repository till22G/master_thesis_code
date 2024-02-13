import os
import json
import numpy as np

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="test_inductivity")
parser.add_argument("--task", default="WN18RR", type=str)
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
entity_dict = None

class EntityDict():
    def __init__(self, path: str, inductive_test_path: str = None) -> None:
        self.entities = []
        self.id2entity = {}
        self.entity2idx = {}
        
        self._load_entity_dict(path, inductive_test_path)
    
    def _load_entity_dict(self, path: str, inductive_test_path: str = None):
        assert os.path.exists(path), "Path is invalid {}:".format(path)
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
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
    
    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]
    
    def get_entity_by_id(self, entity_id: str) -> dict:
        return self.id2entity[entity_id]
    
    def get_entity_by_idx(self, entity_idx: int) -> dict:
        return self.entities[entity_idx]
    
    def __len__(self):
        return len(self.entities)
    

def build_entity_dict():
    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("..", "rebuild_SimKGC", "data", args.task, "entities.json"))
        entity_dict = EntityDict(file_path)
    return entity_dict




def load_training_triples_description_lengths(path: str,
                                       add_forward_triplet: bool = True,
                                       add_backward_triplet: bool = True):
    
    print("Loading entity descriptions for training data")

    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet

    #count = 0
    data = json.load(open(path, 'r', encoding='utf-8'))
    cnt = len(data)
    descriptions = []
    entity_dict = build_entity_dict()
    for i in range(cnt):
        obj = data[i]

        head_id = obj["head_id"]
        head_entity = entity_dict.get_entity_by_id(head_id)
        head_desc = head_entity["entity_desc"]
        head_desc_length = len(head_desc.split())

        #if head_desc_length <= 20 and count < 10:
        #    count += 1
        #    print(head_desc)
        
        tail_id = obj["tail_id"]
        tail_entity = entity_dict.get_entity_by_id(tail_id)
        tail_desc = tail_entity["entity_desc"]
        tail_desc_length = len(tail_desc.split())

        if add_forward_triplet:
            descriptions.append(head_desc_length)
            descriptions.append(tail_desc_length)
        if add_backward_triplet:
            descriptions.append(head_desc_length)
            descriptions.append(tail_desc_length)
    
    return descriptions


print(f"Running analysis for task {args.task}")

entity_dict = build_entity_dict()
desc_len_array = np.empty(len(entity_dict.entities))

for i, entity in enumerate(entity_dict.entities):
    desc = entity["entity_desc"]
    desc_len_array[i] = len(desc.split())


res = np.sum(desc_len_array <= 20)

print(f"{res} out of {len(entity_dict.entities)} entities have a description shorter than 20 words")
print("These are {:.4f} %".format((res / len(entity_dict.entities)) * 100))
print()

data_path = os.path.join("..", "rebuild_SimKGC", "data", args.task, "train.json")
triple_entity_desc_lengths = np.array(load_training_triples_description_lengths(data_path))
res_triples = np.sum(triple_entity_desc_lengths <= 20)

print(f"{res_triples} out of the {len(triple_entity_desc_lengths)} entities in the training data have descriptinos shorter than 20 words")
print("These are {:.4f} %".format((res_triples / len(triple_entity_desc_lengths)) * 100))


