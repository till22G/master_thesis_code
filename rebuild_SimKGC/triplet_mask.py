import torch
import os

from argparser import args
from typing import List
from data_structures import DataPoint, TrainingTripels, EntityDict

script_dir = os.path.dirname(__file__)

training_triples_class = None
entity_dict = None

def construct_triplet_mask(rows: List[DataPoint], cols: List[DataPoint] = None) -> torch.tensor:
    
    global training_triples_class
    if training_triples_class is None:
        file_path = os.path.join(script_dir, os.path.join("..", "data", args.task, "train.json"))
        training_triples_class = TrainingTripels(file_path)

    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("..", "data", args.task, "entities.json"))
        entity_dict = EntityDict(file_path)
        
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
                zero_count += 1
            continue
        for j in range(num_cols):
            if i == j and cols is None: continue 
            if tail_ids_cols[j] in neighbors:
                triplet_mask[i][j] = True

    return triplet_mask

def construct_self_negative_mask(datapoints: List[DataPoint]) -> torch.tensor:
    self_mask = torch.zeros(len(datapoints))
    for i, item in enumerate(datapoints):
        neighbors = training_triples_class.get_neighbors(item.get_head_id(), item.get_relation())
        if item.get_head_id() in neighbors:
            self_mask[i] = 1
    return self_mask.bool()