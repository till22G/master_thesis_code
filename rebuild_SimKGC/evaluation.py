import torch
import os
import json
import torch.utils.data.dataset

from tqdm import tqdm

from data_structures import TrainingTripels, EntityDict, Dataset, DataPoint, collate_fn, load_data
from argparser import args
from logger import logger
from help_functions import move_to_cuda
from evaluation_model import EvaluationModel



def load_all_triples():
    global all_triples
    if all_triples is None:
        file_path_list = [os.path.join(script_dir, os.path.join("data", args.task, "train.json")), \
                          os.path.join(script_dir, os.path.join("data", args.task, "valid.json")), \
                          os.path.join(script_dir, os.path.join("data", args.task, "test.json")) ]
        all_triples = TrainingTripels(file_path_list)
    return all_triples

def build_entity_dict():
    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "entities.json"))
        entity_dict = EntityDict(file_path)
    return entity_dict




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

script_dir = os.path.dirname(__file__)

all_triples = None
entity_dict = None


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
    for _ , batch_dict in enumerate(tqdm(entity_data_loader)):
        #if torch.cuda.is_available():
        #    batch_dict = move_to_cuda(batch_dict)
        batch_dict["only_ent_embedding"] = True
        embedded_entities = eval_model.encode_candidates(batch_dict)
        embedded_entities_list.append(embedded_entities)

    return torch.cat(embedded_entities_list, dim=0)


def get_labels_as_idx(triples):
    labels = []
    entity_dict = build_entity_dict()
    for triple in triples:
        entity_id = triple.get_tail_id()
        entity_idx = entity_dict.entity_to_idx(entity_id)
        labels.append(entity_idx)
    
    return labels


# ignore all known true triples for scoring (filtered setting)
def mask_knows_triples(batch_triples, batch_scores):
    for i in range(len(batch_triples)):
        triple = batch_triples[i]
        
        all_triples = load_all_triples()
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

def eval(model, 
         candidates, 
         forward_triples=True):
    
    print(args.test_path)
    test_data = load_data(path=args.test_path, 
                          add_forward_triplet=forward_triples, 
                          add_backward_triplet=not forward_triples)
    
    hr_embeddings = get_hr_embeddings(model, test_data=test_data)
    labels = get_labels_as_idx(test_data)
    labels = torch.LongTensor(labels).unsqueeze(-1)
    
    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    all_ranks = []
    topk_scores, topk_indices = [], []
    k = 3
    total = hr_embeddings.size(0)
    for i in tqdm(range(0, hr_embeddings.size(0), args.batch_size)):
        step = i + args.batch_size

        # calculate cosine-similarity between hr_embeddings and targets 
        batch_scores = torch.mm(hr_embeddings[i:step,: ], candidates.t())
        batch_labels = labels[i:step].to(batch_scores.device)

        masked_batch_scores = mask_knows_triples(batch_triples=test_data[i:step], batch_scores=batch_scores)
        
        # sort mask results
        _ , sorted_idx = torch.sort(masked_batch_scores, dim=1, descending=True) 
        correct_entities = torch.eq(sorted_idx, batch_labels)
        ranks = torch.nonzero(correct_entities, as_tuple=False)[:,1]

        all_ranks.extend(ranks.tolist())

    hit1 = get_hit_at_k(all_ranks, k = 1)
    hit3 = get_hit_at_k(all_ranks, k = 3)
    hit10 = get_hit_at_k(all_ranks, k = 10)
    mean_rank = sum(all_ranks) / len(all_ranks) + 1
    mrr = sum([1/(item +1) for item in all_ranks]) / len(all_ranks)

    print(hit1)
    print(hit3)
    print(hit10)
    print(mean_rank)
    print(mrr)



def main():
    train_path = args.train_path
    test_path = args.test_path

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # build model and load checkpoint
    eval_model = EvaluationModel()
    eval_model.load_checkpoint(checkpoint_path=args.eval_model_path)
    
    entity_dict = build_entity_dict()
    entity_embeddings = get_entity_embeddings(entity_dict=entity_dict, eval_model=eval_model)

    results_forward = eval(model=eval_model,
                           candidates=entity_embeddings,
                           forward_triples=True)
    results_backward = eval(model=eval_model, 
                            candidates=entity_embeddings,
                            forward_triples=False)
    

            
if __name__ == "__main__":
    main()