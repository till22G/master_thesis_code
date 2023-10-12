import torch

from argparser import args
from typing import List
from data_structures import DataPoint, TrainingTripels, EntityDict

training_triples_class = TrainingTripels("../data/FB15k237/train.json")
entity_dict = EntityDict("../data/FB15k237/entities.json")

def construct_triplet_mask(rows: List[DataPoint], cols: List = None) -> torch.tensor:
    num_rows = len(rows)
    num_cols = num_rows if cols is None else len(cols)
            
    tail_ids_rows = torch.LongTensor([entity_dict.entity_to_idx(datapoint.get_tail_id()) for datapoint in rows])
    tail_ids_cols = tail_ids_rows if cols is None \
        else torch.LongTensor([entity_dict.entity_to_idx(datapoint.get_tail_id()) for datapoint in cols])
    
    triplet_mask = tail_ids_rows.unsqueeze(1) == tail_ids_cols.unsqueeze(0)
    if cols is None:
        triplet_mask.fill_diagonal_(False)
        
    zero_count = 0
    # mask out neighbors
    for i in range(num_rows):
        head, relation = rows[i].get_head_id(), rows[i].get_relation()
        neighbors = training_triples_class.get_neighbors(head, relation)
        if len(neighbors) <= 1:
            if len(neighbors) == 0:
                if i == 1: print(head, relation)
                zero_count += 1
            continue
        for j in range(num_cols):
            if i == j and cols is None: continue 
            if tail_ids_cols[j] in neighbors:
                triplet_mask[i][j] = True

    return triplet_mask