import torch
import os

from typing import List

#from config import args
from argparser import args
#from triplet import EntityDict
from data_structures import EntityDict, DataPoint, NeighborhoodGraph
#from dict_hub import get_link_graph
#from doc import Example
neigborhood_graph = None
entity_dict = None

script_dir = os.path.dirname(__file__)

def build_neighborhood_graph():
    global neigborhood_graph
    if not neigborhood_graph:
        entity_dict = build_entity_dict()
        neigborhood_graph = NeighborhoodGraph(args.train_path, entity_dict=entity_dict)
    return neigborhood_graph

def build_entity_dict():
    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "entities.json"))
        entity_dict = EntityDict(file_path)
    return entity_dict


def rerank_by_graph(batch_score: torch.tensor,
                    examples: List[DataPoint],
                    entity_dict: EntityDict):

    if args.task == 'wiki5m_ind':
        assert args.neighbor_weight < 1e-6, 'Inductive setting can not use re-rank strategy'

    if args.neighbor_weight < 1e-6:
        return

    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        neigborhood_graph = build_neighborhood_graph()
        n_hop_indices = neigborhood_graph.get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=args.rerank_n_hop)
        delta = torch.tensor([args.neighbor_weight for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)

        # The test set of FB15k237 removes triples that are connected in train set,
        # so any two entities that are connected in train set will not appear in test,
        # however, this is not a trick that could generalize.
        # by default, we do not use this piece of code .

        # if args.task == 'FB15k237':
        #     n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
        #                                                               entity_dict=entity_dict,
        #                                                               n_hop=1)
        #     n_hop_indices.remove(entity_dict.entity_to_idx(cur_ex.head_id))
        #     delta = torch.tensor([-0.5 for _ in n_hop_indices]).to(batch_score.device)
        #     n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)
        #
        #     batch_score[idx].index_add_(0, n_hop_indices, delta)
