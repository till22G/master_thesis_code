import os
from data_structures import Dataset, DataPoint

file_path = "../data/FB15k237/train.json"
assert os.path.exists(file_path), "Invalid path"

dataset = Dataset(file_path, task=None)

for i in [0, 1]:
    print(dataset[i].head_id)
    print(dataset[i].head_desc)
    print(dataset[i].relation)
    print(dataset[i].tail_id)
    print(dataset[i].tail_desc)