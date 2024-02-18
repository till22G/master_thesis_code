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
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(prog="evaluate graph density and token length")
parser.add_argument("--task", default="WN18RR", type=str)
parser.add_argument("--max-context-size", default=0, type=int)
parser.add_argument("--cutoff", default=512, type=int)
args = parser.parse_args()

task = args.task
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


def format_y_ticks(value, _):
    return f'{value / 1000:.0f}K'

def print_hist_num_neighbors(data, bin_size=1):
    print("Creating histogram ...")
    plt.figure(figsize=(8, 4))
    # Apply the formatter to the y-axis
    #plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))
    plt.hist(data, bins=range(min(data), max(data) + bin_size, bin_size), color="black", edgecolor='black')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Frequency')
    if args.task == "wiki5m_trans":
        dataset = "Wikidata5m"
    else:
        dataset = task
    plt.title(f'Histogram number of neighbors for {dataset}')

    try:
        save_path = f"../plots/plots/histogram_num_of_neighbors_{task}"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=600)
        plt.clf()
        print(f"Saved histrogram to {save_path}")
    
    except Exception as e:
        print(f"Could not save histogram to: {save_path}")


path = os.path.join('..', 'plots', 'plot_data', f"graph_density_{task}.npy")
if os.path.exists(path):
    print(f"Loading graph density information from: {path}")
    node_number_of_neighbors = np.load(path)

else:
    if not os.path.exists(path):
        print(f"No exisiting data found for: {path}")
    
        entity_dict = get_entity_dict()
        args.max_context_size = len(entity_dict.entities)
        neighborhood_graph = get_hop_1_graph_new()

        node_number_of_neighbors = []
        for entity in tqdm(entity_dict.entities):
            neighbors = neighborhood_graph.get_neighbors(entity["entity_id"])
            node_number_of_neighbors.append(len(neighbors))

    try:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.save(path, node_number_of_neighbors)
    except Exception as e:
        print(f"Could not save data to: {path}")

neighbors_median = np.median(node_number_of_neighbors)
neighbors_avg = np.average(node_number_of_neighbors)
    
token_output_dict = {"dataset" : task,
                     "median" : neighbors_median,
                     "average" : neighbors_avg}   

neighbor_file_path = os.path.join('..', 'plots', 'reports', f'report_graph_density_{task}.json')

with open(neighbor_file_path, "w") as json_file:
    json.dump(token_output_dict, json_file)

print(f"Task: {args.task}")
print(f"median for number of neighbors: {neighbors_median}")
print("Avg values for number of neighbors: {:.4f}".format(neighbors_avg))
print(f"Max number of neighbors: {np.max(node_number_of_neighbors)}")

# set a largest bin
max_number = args.cutoff
cut_node_number_of_neighbors = node_number_of_neighbors[node_number_of_neighbors > max_number] = max_number
print_hist_num_neighbors(node_number_of_neighbors)