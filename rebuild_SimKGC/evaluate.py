import os
import json
import tqdm
import torch
import glob

from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict

from argparser import args
#from doc import load_data, Example
from data_structures import load_data, DataPoint, EntityDict, TrainingTripels, Dataset, collate_fn
from predict import BertPredictor
#from dict_hub import get_entity_dict, get_all_triplet_dict
#from triplet import EntityDict
from rerank import rerank_by_graph
from logger import logger
from help_functions import move_to_cuda
from evaluation_model import EvaluationModel



def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(path=os.path.join(os.path.join("data", args.task, "entities.json")),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()

def get_entity_dict():
    entity_dict = EntityDict(os.path.join(os.path.join("data", args.task, "entities.json")))
    return entity_dict

def get_all_triplet_dict():
    path_list = [os.path.join("data", args.task, set + ".json") for set in ["train", "valid", "test"]]
    all_triplet_dict = TrainingTripels(path_list)
    return all_triplet_dict

entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()
neigborhood_graph = None

@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


def get_hr_embeddings(eval_model, test_data):

    test_dataset = Dataset(path="", data_points=test_data)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers
        )
    
    embedded_hr_list = []
    for i, batch_dict in enumerate((test_data_loader)):
        if torch.cuda.is_available():
            batch_dict = move_to_cuda(batch_dict)
        embedded_hr = eval_model.encode_hr(batch_dict)
        embedded_hr_list.append(embedded_hr)

    return torch.cat(embedded_hr_list, dim=0)

def get_entity_embeddings(entity_dict, eval_model):
    entity_datapoints = []
    for entity in entity_dict.entities:
        datapoint = DataPoint(tail_id=entity["entity_id"],
                              tail=entity["entity"],
                              tail_desc=entity["entity_desc"])
        entity_datapoints.append(datapoint)

    entity_data_set = Dataset(path="", data_points=entity_datapoints)

    entity_data_loader = torch.utils.data.DataLoader(
        entity_data_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers
        )
    
    embedded_entities_list = []
    for _ , batch_dict in enumerate(tqdm.tqdm(entity_data_loader)):
        #if torch.cuda.is_available():
        #    batch_dict = move_to_cuda(batch_dict)
        batch_dict["only_ent_embedding"] = True
        embedded_entities = eval_model.encode_candidates(batch_dict)
        embedded_entities_list.append(embedded_entities)

    return torch.cat(embedded_entities_list, dim=0)


def get_labels_as_idx(triples):
    labels = []
    for triple in triples:
        entity_id = triple.get_tail_id()
        entity_idx = entity_dict.entity_to_idx(entity_id)
        labels.append(entity_idx)
    
    return labels


# ignore all known true triples for scoring (filtered setting)
def mask_knows_triples(batch_triples, batch_scores):
    for i in range(len(batch_triples)):
        triple = batch_triples[i]
        
        all_triples = all_triplet_dict
        neighbors = all_triples.get_neighbors(triple.get_head_id(), triple.get_relation())
        tail_id = triple.get_tail_id() 
        mask_idx = [entity_dict.entity_to_idx(entity_id) for entity_id in neighbors if entity_id != tail_id]
        mask_idx = torch.LongTensor(mask_idx).to(batch_scores.device)
        batch_scores[i].index_fill_(0, mask_idx, -1)
    
    return batch_scores
        
def get_hit_at_k(ranks, k=1):
    hits = 0
    k = k-1
    for rank in ranks:
        if rank <= k:
            hits += 1
    return hits / len(ranks)


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[DataPoint],
                    k=3, batch_size=256) -> Tuple:

    """ target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []
    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    all_ranks = []
    topk_scores, topk_indices = [], []
    k = 3
    total = hr_tensor.size(0)
    for i in tqdm.tqdm(range(0, hr_tensor.size(0), args.batch_size)):
        step = i + args.batch_size

        # calculate cosine-similarity between hr_embeddings and targets 
        batch_scores = torch.mm(hr_tensor[i:step,: ], entities_tensor.t())
        batch_labels = target[i:step].to(batch_scores.device)

        masked_batch_scores = mask_knows_triples(batch_triples=examples[i:step], batch_scores=batch_scores)
        
        # sort mask results
        sorted_scores , sorted_idx = torch.sort(masked_batch_scores, dim=1, descending=True) 
        correct_entities = torch.eq(sorted_idx, batch_labels)
        ranks = torch.nonzero(correct_entities, as_tuple=False)[:,1]

        all_ranks.extend(ranks.tolist())

    hit1 = get_hit_at_k(all_ranks, k = 1)
    hit3 = get_hit_at_k(all_ranks, k = 3)
    hit10 = get_hit_at_k(all_ranks, k = 10)
    mean_rank = sum(all_ranks) / len(all_ranks) + 1
    mrr = sum([1/(item +1) for item in all_ranks]) / len(all_ranks)

    topk_scores.extend(sorted_scores[:, :k].tolist())
    topk_indices.extend(sorted_idx[:, :k].tolist())

    metrics = {'mean_rank': round(mean_rank, 4), 'mrr': round(mrr, 4), 'hit@1': round(hit1, 4), 'hit@3': round(hit3, 4), 'hit@10': round(hit10, 4)} """


    ## top k-scroes is reported wrongly
    return topk_scores, topk_indices, metrics, ranks


def predict_by_split():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)

    #predictor = BertPredictor()
    #predictor.load(ckt_path=args.eval_model_path)
    predictor = EvaluationModel()
    predictor.load_checkpoint(args.eval_model_path)
    entity_tensor = get_entity_embeddings(entity_dict=entity_dict, eval_model=predictor)
    
    #entity_tensor = predictor.predict_by_entities(entity_dict.entities)

    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))


def eval_single_direction(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)


    hr_tensor = get_hr_embeddings(predictor, examples)

    #hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    #target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    target = get_labels_as_idx(triples=examples)
    logger.info('predict tensor done, compute metrics...')

    """ topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                target=target, examples=examples,
                                                                batch_size=batch_size) """
    
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []
    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    all_ranks = []
    topk_scores, topk_indices = [], []
    k = 3
    total = hr_tensor.size(0)
    for i in tqdm.tqdm(range(0, hr_tensor.size(0), args.batch_size)):
        step = i + args.batch_size

        # calculate cosine-similarity between hr_embeddings and targets 
        batch_scores = torch.mm(hr_tensor[i:step,: ], entity_tensor.t())
        batch_labels = target[i:step].to(batch_scores.device)

        masked_batch_scores = mask_knows_triples(batch_triples=examples[i:step], batch_scores=batch_scores)
        
        # sort mask results
        sorted_scores , sorted_idx = torch.sort(masked_batch_scores, dim=1, descending=True) 
        correct_entities = torch.eq(sorted_idx, batch_labels)
        ranks = torch.nonzero(correct_entities, as_tuple=False)[:,1]

        all_ranks.extend(ranks.tolist())

    hit1 = get_hit_at_k(all_ranks, k = 1)
    hit3 = get_hit_at_k(all_ranks, k = 3)
    hit10 = get_hit_at_k(all_ranks, k = 10)
    mean_rank = sum(all_ranks) / len(all_ranks) + 1
    mrr = sum([1/(item +1) for item in all_ranks]) / len(all_ranks)

    topk_scores.extend(sorted_scores[:, :k].tolist())
    topk_indices.extend(sorted_idx[:, :k].tolist())

    metrics = {'mean_rank': round(mean_rank, 4), 'mrr': round(mrr, 4), 'hit@1': round(hit1, 4), 'hit@3': round(hit3, 4), 'hit@10': round(hit10, 4)}
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

   
    return metrics


if __name__ == '__main__':
    predict_by_split()
