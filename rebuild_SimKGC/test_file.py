
import os

from data_structures import Dataset, DataPoint, EntityDict, TrainingTripels

import torch.utils.data

from typing import List
from data_structures import _concat_name_desciption

from argparser import args
from  data_structures import collate_fn

from tqdm import tqdm
from model import CustomModel
import time

path = "../data/FB15k237/entities.json"
entity_dict = EntityDict(path)
entity_id = entity_dict.get_entity_by_idx(5643)["entity_id"]
entity_idx = entity_dict.entity_to_idx(entity_id)


training_triples_class = TrainingTripels("../data/FB15k237/train.json")
print(training_triples_class.get_neighbors("/m/04w391", "inverse participant popstra base dated celebrity popstra base"))
print()

file_path = "../data/FB15k237/train.json"
assert os.path.exists(file_path), "Invalid path"

dataset = Dataset(file_path)

 
train_data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=30,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True
)

model = CustomModel(args)
count = 0
for i, item in enumerate(tqdm(train_data_loader)):
    out = model(**item)
    model.compute_logits(encodings=out, batch_data=item)
    
    count += 1
    if count == 1:
        break

""" t1 = "head"
t2 = "head desc"
t2 = None

t = _concat_name_desciption(t1, t2) 
print(t)
"""
