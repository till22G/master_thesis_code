import os
import json
from typing import List
from collections import deque, OrderedDict
from dataclasses import dataclass
import time 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics 
from transformers import AutoTokenizer
from typing import Optional
import argparse

parser = argparse.ArgumentParser(prog="evaluate graph density and token length")
parser.add_argument("--task", default="WN18RR", type=str)
parser.add_argument("--use-context-relation", action="store_true")
parser.add_argument("--max-num-desc-tokens", default=50, type=int)
parser.add_argument("--max-context-size", default=10000, type=int)
parser.add_argument("--max-number-tokens", default=10000, type=int)
parser.add_argument("--use-context-descriptions", action="store_true")
parser.add_argument("--cutoff", default=512)
args = parser.parse_args()

task = args.task

args.use_context_relation = False
script_dir = os.path.dirname(__file__)


entity_dict = None
tokenizer = None
training_triples = None
hop_1_graph_new = None
entities = {}

data_dir = os.path.join(os.path.dirname(script_dir), "rebuild_SimKGC", "data")

def get_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(path=os.path.join(data_dir, task, "entities.json"))
    return entity_dict

def get_training_triples():
    global training_triples
    if not training_triples:
        training_triples = load_data(path=os.path.join(data_dir, task, "train.json"), add_backward_triplet=True)
    return training_triples 

def get_hop_1_graph_new():
    global hop_1_graph_new
    if not hop_1_graph_new:
        hop_1_graph_new = Hop1IndexNew(train_path=os.path.join(data_dir, task, "train.json"),
                                       entity_dict=get_entity_dict(),
                                       max_context_size=args.max_context_size)
    return hop_1_graph_new

def build_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    return tokenizer

class EntityDict():
    def __init__(self, path: str, inductive_test_path: str = None) -> None:
        self.entities = []
        self.id2entity = {}
        self.entity2idx = {}
        
        self._load_entity_dict(path, inductive_test_path)
    
    def _load_entity_dict(self, path: str, inductive_test_path: str = None):
        assert os.path.exists(path), "Path is invalid {}:".format(path)
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
        print("Loading entity dict ...")
        with open(path, "r", encoding="utf8") as infile:
            data = json.load(infile)
            
        self.entities = data

        if inductive_test_path:
            with open(inductive_test_path, "r", encoding="utf-8") as infile:
                data = json.load(infile)

            entity_ids = set()
            for triple in data:
                entity_ids.add(triple['head_id'])
                entity_ids.add(triple['tail_id'])
            self.entities = [entity for entity in self.entities if entity["entity_id"] in entity_ids]

        self.id2entity = {entity["entity_id"]: entity for entity in data}
        self.entity2idx = {entity["entity_id"]: i for i, entity in enumerate(self.entities)}

        print("Entity dict successfully loaded")
    
    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]
    
    def get_entity_by_id(self, entity_id: str) -> dict:
        return self.id2entity[entity_id]
    
    def get_entity_by_idx(self, entity_idx: int) -> dict:
        return self.entities[entity_idx]
    
    def __len__(self):
        return len(self.entities)


class DataPoint():
    def __init__(self, 
                 head_id: str = None,
                 head: str = None,
                 head_desc: str = None,
                 relation: str = None, 
                 tail_id: str = None, 
                 tail: str = None,
                 tail_desc: str = None) -> None: 

        self.head_id = head_id
        self.head = head
        self.head_desc = head_desc
        self.relation = relation
        self.tail_id = tail_id
        self.tail = tail
        self.tail_desc = tail_desc
    
    def get_head_id(self) -> str:
        if self.head_id is not None:
            return self.head_id
        else:
            return ""
    
    def get_head(self) -> str:
        if self.head is not None:
            return self.head
        else:
            return ""
        
    def get_head_desc(self) -> str:
        if self.head_desc is not None:
            return self.head_desc
        else:
            return ""
        
    def get_relation(self) -> str:
        if self.relation is not None:
            return self.relation
        else:
            return ""
    
    def get_tail_id(self) -> str:
        if self.tail_id is not None:
            return self.tail_id
        else:
            return ""
        
    def get_tail(self) -> str:
        if self.tail is not None:
            return self.tail
        else:
            return ""
        
    def get_tail_desc(self) -> str:
        if self.tail is not None:
            return self.tail_desc
        else:
            return ""
    

def load_entities(path) -> None:
    global entities
    
    assert os.path.exists(path), "Path is invalid: {}".format(path)
    assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
    with open (path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
        
    for item in data:
        entities[item["entity_id"]] = item
    


def load_data(path: str, add_forward_triplet: bool = True, add_backward_triplet: bool = True) -> List[DataPoint]:
        global entities
        if not entities:
            load_entities(os.path.join(data_dir, task ,"entities.json"))
        
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
            
        datapoints = []
        for item in data:
            if add_forward_triplet:
                datapoints.append(DataPoint(item["head_id"],
                                            item["head"],
                                            entities[item["head_id"]].get("entity_desc", ""),
                                            item["relation"], 
                                            item["tail_id"],
                                            item["tail"],
                                            entities[item["tail_id"]].get("entity_desc", "")
                                            ))
            if add_backward_triplet:
                datapoints.append(DataPoint(item["tail_id"],
                                            item["tail"],
                                            entities[item["tail_id"]].get("entity_desc", ""),
                                            "inverse {}".format(item["relation"]), 
                                            item["head_id"],
                                            item["head"],
                                            entities[item["head_id"]].get("entity_desc", "")
                                            ))
            
        return datapoints


def triples_to_np(triples: List[DataPoint]) -> np.ndarray:
    np_triples = np.empty((len(triples), 3), dtype=object)
    
    for i, triple in enumerate(triples):
        head_inx = entity_dict.entity_to_idx(triple.head_id)
        np_triples[i] = [head_inx, triple.relation, triple.tail]

    return np_triples


class Hop1IndexNew:
    def __init__(self, train_path, entity_dict, key_col=0, max_context_size=10, shuffle=False):
        print("Building neighborhood graph ...")
        self.num_entities = len(entity_dict)
        triples = json.load(open(train_path, 'r', encoding='utf-8'))
        num_triples = len(triples)
        np_triples = np.empty((num_triples * 2, 3), dtype=object)

        for i, triple in enumerate(triples):
            head_idx = entity_dict.entity_to_idx(triple["head_id"])
            np_triples[i] = [head_idx, triple["relation"], triple["tail_id"]]
            # add reverse triples
            tail_idx = entity_dict.entity_to_idx(triple["tail_id"])
            np_triples[num_triples + i] = [tail_idx, "inverse " + triple["relation"], triple["head_id"]]
            
        triples = np_triples

        self.max_context_size = max_context_size
        self.shuffle = shuffle
        self.triples = np.copy(triples[triples[:, key_col].argsort()])
        keys, values_offset = np.unique(
            self.triples[:, key_col].astype(int), axis=0, return_index=True)
        
        values_offset = np.append(values_offset, len(self.triples))
        self.keys = keys
        self.values_offset = values_offset
        self.key_to_start = np.full([self.num_entities,], -1)
        self.key_to_start[keys] = self.values_offset[:-1]
        self.key_to_end = np.full([self.num_entities,], -1)
        self.key_to_end[keys] = self.values_offset[1:]

        print("Neighborhood graph successfully build")

    def __getitem__(self, item):
        start = self.key_to_start[item]
        end = self.key_to_end[item]
        context = self.triples[start:end, [1, 2]]
        if self.shuffle:
            context = np.copy(context)
            np.random.shuffle(context)
        if end - start > self.max_context_size: 
            context = context[:self.max_context_size]
        return context

    def get_neighbors(self, item):
        if item == '':
            return set()
        entity_dict = get_entity_dict()
        idx = entity_dict.entity_to_idx(item)
        return self[idx]

    def get(self, item):
        return self[item]    
    
    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 max_neighbors: int = 100000) -> set:
        
        if n_hop < 1:
            return set()
        
        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                n_id = queue.popleft()
                neighbors = self.get_neighbors(n_id)
                neighbor_ids = [neighbor[1] for neighbor in neighbors]
                for neighbor_id in neighbor_ids:
                    if neighbor_id not in seen_eids:
                        queue.append(neighbor_id)
                        seen_eids.add(neighbor_id)
                        if len(seen_eids) > max_neighbors:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity is None:
        entity = ""
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def _build_context_string(head_id: str, relation: str = None, tail_id: str = None, use_context_descriptions: bool = False):
    context_string = ""
    if head_id == "":
        return ""
    entity_dict = get_entity_dict()
    head_idx = entity_dict.entity_to_idx(head_id)
    context = get_hop_1_graph_new().get(head_idx)

    for neighbor in context:
        n_relation, neighbor_id = neighbor
        n_text = entity_dict.get_entity_by_id(neighbor_id)["entity"]
        if args.use_context_descriptions:
            n_tail_desc = ' '.join(entity_dict.get_entity_by_id(neighbor_id)["entity_desc"].split()[:40])
            n_text = _concat_name_desc(n_text, n_tail_desc)
        if args.use_context_relation:
            context_string += f", {n_relation} {n_text}"
        else:
            context_string += f", {n_text}"
        if context_string == ", ":
            return ""
    return f"{context_string[2:]}"


def _tokenize(text1: str, text2: Optional[str] = None, text_pair: Optional[str] = None ) -> dict:
    tokenizer = build_tokenizer() 
    
    text1_encodings = tokenizer.encode(text1, max_length=args.max_num_desc_tokens, truncation=True, add_special_tokens=False)
    text1 = tokenizer.decode(text1_encodings)

    if text2:
        text = text1 + " : " + text2
    else:
        text = text1

    encodings = tokenizer(text=text,
                          text_pair=text_pair if text_pair else None,
                          add_special_tokens=True,
                          max_length=10000,
                          return_token_type_ids=False,
                          truncation=True
                        )

    return encodings

def print_hist_of_num_tokens(data, bin_size=1):
    
    #percentile_95_value = np.percentile(data_total, 95)
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=range(min(data), max(data) + bin_size, bin_size), edgecolor='black')
    #plt.axvline(x=percentile_95_value, color='green', linestyle='--', label='95th Percentile')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')

    if args.task == "wiki5m_trans":
        dataset = "Wikidata5m"
    else:
        dataset = task
    plt.title(f'Number of tokens for verbalized entity description + context {dataset}')
    
    save_path = f"../plots/plots/histogram_num_tokens_{task}_context_size_{args.max_context_size}_relations_{args.use_context_relation}"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    plt.clf()



path = os.path.join('..', 'plots', 'plot_data', f"token_length_{task}_max_context_size_{args.max_context_size}_use_rel_{args.use_context_relation}_max_tokens_{args.max_number_tokens}.npy")
if os.path.exists(path):
    print(f"Loading graph density information from: {path}")
    token_lenght_list = np.load(path)

else:
    if not os.path.exists(path):
        print(f"No exisiting data found for: {path}")

    entity_dict = get_entity_dict()
    token_lenght_list = []
    for entity in tqdm(entity_dict.entities):
        entity_id = entity["entity_id"]
        entity_name = entity["entity"]
        desc = entity["entity_desc"]

        textualaized_entity = _concat_name_desc(entity_name, desc)
        context_string = _build_context_string(entity_id)
        
        input_sequence = _tokenize(textualaized_entity, context_string)
        token_len = len(input_sequence["input_ids"])
        token_lenght_list.append(token_len)

    try:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.save(path, token_lenght_list)
        print(f"Histogram saved to: {path}")
    except Exception as e:
        print(f"Could not save data to: {path}")
    
token_lenght_list = np.array(token_lenght_list)
token_len_median = np.median(token_lenght_list)
token_len_avg = np.average(token_lenght_list)
    
token_output_dict = {"dataset" : task,
                     "median" : token_len_median,
                     "average" : token_len_avg}


neighbor_file_path = os.path.join('..', 'plots', 'reports', f"token_length_{task}_max_context_size_{args.max_context_size}_use_rel_{args.use_context_relation}.json")

with open(neighbor_file_path, "w") as json_file:
    json.dump(token_output_dict, json_file)

print(f"Task: {args.task}")
print(f"median for number of tokens: {token_len_median}")
print("Average number of tokens: {:.4f}".format(token_len_avg))
print(f"Max number of tokens: {np.max(token_lenght_list)}")

result = np.percentile(token_lenght_list, [25,50,75, 90, 95, 99])
for p, value in zip( [25,50,75, 90, 95, 99], result):
    print(f"{p}th percentile: {value}")

cutoff_tokens = 512
percentile_rank = np.sum(token_lenght_list <= cutoff_tokens) / len(token_lenght_list) * 100
print("Percentile rank for cutoff token at {} tokens: {:.4f} ".format(cutoff_tokens, percentile_rank))

cut_node_number_of_neighbors = token_lenght_list[token_lenght_list > args.cutoff] = args.cutoff
print_hist_of_num_tokens(token_lenght_list)








