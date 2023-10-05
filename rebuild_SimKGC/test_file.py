
import os

from data_structures import Dataset, DataPoint

import torch.utils.data

from typing import List
from data_structures import _concat_name_desciption

from argparser import args
from  data_structures import collate_fn

file_path = "../data/FB15k237/train.json"
assert os.path.exists(file_path), "Invalid path"

dataset = Dataset(file_path)

 
train_data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True
)

for i, item in enumerate(train_data_loader):
    if i == 0:
        print("item")
        print(item)



""" 
data_point = dataset[92]
print(data_point.get_head_id())
print(data_point.get_head())
print(data_point.get_head_desc())
print(data_point.get_relation())
print(data_point.get_tail_id())
print(data_point.get_tail())
print(data_point.get_tail_desc())
 """
 
""" t1 = "head"
t2 = "head desc"
t2 = None

t = _concat_name_desciption(t1, t2) 
print(t)
"""
