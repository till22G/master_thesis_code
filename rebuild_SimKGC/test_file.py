import os
from data_structures import Dataset, DataPoint

file_path = "../data/FB15k237/train.json"
assert os.path.exists(file_path), "Invalid path"

dataset = Dataset(file_path, task=None)