import torch

from argparser import args
from typing import List
from data_structures import DataPoint

def construct_triplet_mask(rows: List[DataPoint], cols: List = None) -> torch.tensor:
    num_rows = len(rows)
    if cols is None:
        num_cols = num_rows 
    else:
        num_rows = len(cols)
        
    tail_ids_rows = torch.LongTensor([datapoint["tail_id"] for datapoint in rows])
    tail_ids_cols = tail_ids_rows if cols is None else torch.LongTensor([datapoint["tail_id"] for datapoint in cols])
    
    triplet_mask = tail_ids_rows.unsqueeze(1) == tail_ids_cols.unsqueeze(0)
    if cols is None:
        triplet_mask.fill_diagonal_(False)
        
    # maks out neighbors
    
    
    return triplet_mask