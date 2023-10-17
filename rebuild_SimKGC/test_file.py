
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
from trainer import CustomTrainer


args.train_path = "../data/FB15k237/train.json"
args.valid_path = "../data/FB15k237/valid.json"
trainer = CustomTrainer(args)
trainer.training_loop()



""" 
file_path = "../data/FB15k237/train.json"
assert os.path.exists(file_path), "Invalid path"

dataset = Dataset(file_path)

 
train_data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=10,
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
    if count == 3:
        break """

""" t1 = "head"
t2 = "head desc"
t2 = None

t = _concat_name_desciption(t1, t2) 
print(t)
"""
