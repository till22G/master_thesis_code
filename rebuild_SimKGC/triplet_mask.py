import torch

from argparser import args
from typing import List
from data_structures import DataPoint, TrainingTripels

training_triples_class = TrainingTripels("../data/FB15k237/train.json")

def construct_triplet_mask(rows: List[DataPoint], cols: List = None) -> torch.tensor:
    num_rows = len(rows)
    num_cols = num_rows if cols is None else len(cols)
        
    tail_ids_rows = torch.LongTensor([datapoint["tail_id"] for datapoint in rows])
    tail_ids_cols = tail_ids_rows if cols is None else torch.LongTensor([datapoint["tail_id"] for datapoint in cols])
    
    triplet_mask = tail_ids_rows.unsqueeze(1) == tail_ids_cols.unsqueeze(0)
    if cols is None:
        triplet_mask.fill_diagonal_(False)
        
    # maks out neighbors
    for i in range(num_rows):
        head, relation = rows[i].get_head(), rows[i].get_relation()
        neighbors = training_triples_class.get_neighbors(head, relation)
        if len(neighbors) <= 1: continue
        for j in range(num_cols):
            if i == j and cols is None: continue 
            if tail_ids_cols[j] in neighbors:
                triplet_mask[i][j] = True
    
    return triplet_mask