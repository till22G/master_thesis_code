import os
import json
import torch

from torch.utils.data import Dataset
from typing import List, Optional
from transformers import AutoTokenizer
from collections import deque

#from triplet_mask import construct_self_negative_mask, construct_triplet_mask
from logger import logger
from argparser import args

script_dir = os.path.dirname(__file__)

tokenizer: AutoTokenizer = None

entities = {}
neigborhood_graph = None
training_triples_class = None

class EntityDict():
    def __init__(self, path) -> None:
        self.entities = []
        self.id2entity = {}
        self.entity2idx = {}
        
        self._load_entity_dict(path)
    
    def _load_entity_dict(self, path):
        assert os.path.exists(path), "Path is invalid {}:".format(path)
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
        
        with open(path, "r", encoding="utf8") as infile:
            data = json.load(infile)
            
        self.entities = data    
        self.id2entity = {entity["entity_id"]: entity for entity in data}
        self.entity2idx = {entity["entity_id"]: i for i, entity in enumerate(data)}
    
    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]
    
    def get_entity_by_id(self, entity_id: str) -> dict:
        return self.id2entity[entity_id]
    
    def get_entity_by_idx(self, entity_idx: int) -> dict:
        return self.entities[entity_idx]
    
    def __len__(self):
        return len(self.entities)


class TrainingTripels():
    def __init__(self, path_list) -> None:
        self.training_triples = []
        self.hr2tails = {}

        for path in path_list:
            self._load_training_tripels(path)
        
    def _load_training_tripels(self, path) -> None:
        
        assert os.path.exists(path), "Path is invalid {path}"
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
        
        #logger.info("Loading training triples from {}".format(path))
        
        with open(path, "r", encoding="utf-8") as inflile:
            data = json.load(inflile)
                
        for item in data:
            self.training_triples.append(item)
            #############################################################
            if True:
            #################################################################
                inv_item = {"head_id": item["tail_id"],
                            "head" : item["tail"],
                            "relation": " ".join(("inverse", item["relation"])),
                            "tail_id": item["head_id"],
                            "tail" : item["head"]
                }
                self.training_triples.append(inv_item)
                
        for item in self.training_triples:
            key = (item["head_id"], item["relation"])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(item["tail_id"])

    def get_neighbors(self, head_id: str, relation: str) -> set:
        return self.hr2tails.get((head_id, relation), set())
    
    def get_triplet(self, idx: int) -> dict:
        return self.training_triples[idx]
    
    def get_triplet_list(self) -> list: 
        return self.training_triples
        

class NeighborhoodGraph():
    def __init__(self, path) -> None:
        self.graph = {}
        #logger.info("Building neighborhood graph from {}".format(path))
            
        global training_triples_class
        if training_triples_class is None:
            training_triples_class = TrainingTripels([path])
            
        for item in training_triples_class.get_triplet_list():
            if item["head_id"] not in self.graph:
                self.graph[item["head_id"]] = set()
            self.graph[item["head_id"]].add(item["tail_id"])
           
            if item["tail_id"] not in self.graph:
                self.graph[item["tail_id"]] = set()
            self.graph[item["tail_id"]].add(item["head_id"])
        
        #logger.info("Neighborhood graph succesfully build")
            
        
    def get_neighbors(self, entity_id: str, num_neigbhours: int = 10):
        neigbours = sorted(self.graph.get(entity_id, set()))
        return neigbours[:num_neigbhours]
    
    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])
    

def build_neighborhood_graph():
    global neigborhood_graph
    if neigborhood_graph == None:
        neigborhood_graph = NeighborhoodGraph(args.train_path)
    return neigborhood_graph
    
    
    
    
def load_entities(path) -> None:
    
    assert os.path.exists(path), "Path is invalid: {}".format(path)
    assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"

    logger.info("Loading entity descriptions from {}".format(path))
    
    with open (path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
            
    for item in data:
        entities[item["entity_id"]] = item
        
    logger.info("{} entity descriptinos loaded".format(len(entities)))
    

def _concat_name_desciption(entity_head: str, entity_desc: str):
    if not entity_desc:
        return entity_head
    if entity_desc.startswith(entity_head):
        entity_desc = entity_desc[len(entity_head):].strip()
    return "{}: {}".format(entity_head, entity_desc)


def _tokenize_text(text:str, relation: Optional[str] = None) -> dict:
    global tokenizer
    if tokenizer is None:
        create_tokenizer()
    
    tokens = tokenizer(text,
                       text_pair = relation,
                       add_special_tokens = True,
                       truncation=True,
                       max_length = args.max_number_tokens,
                       return_token_type_ids = True)
    
    return tokens


def create_tokenizer():
    global tokenizer
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        #logger.info("Created tokenizer from {}".format(args.pretrained_model))
    return tokenizer

    
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
        
    def encode_to_dict(self) -> dict:
        
        head_desc, tail_desc = self.head_desc, self.tail_desc
        
        # still need to code what happens when this is set to true
        if args.use_neighbors:
            # the entity names should be padded with information from neighbour if
            # their own description is to short
            if len(head_desc.split()) < 20:
                head_desc = " ".join((head_desc, add_neighbor_names(self.head_id, self.tail_id)))
            if len(tail_desc.split()) < 20:
                tail_desc = " ".join((tail_desc, add_neighbor_names(self.tail_id, self.head_id)))
                  
        head_text = _concat_name_desciption(self.get_head(), head_desc)
        hr_tokens = _tokenize_text(text=head_text, relation=self.relation)
        
        tail_text = _concat_name_desciption(self.get_tail(), tail_desc)
        t_tokens = _tokenize_text(tail_text)
        
        h_tokens = _tokenize_text(head_text)

        """ print("------------ head-rel decoded -----------------")
        decoded_hr_tokens =  create_tokenizer().decode(hr_tokens["input_ids"])
        print(decoded_hr_tokens)
        print("------------ tail decoded -----------------")
        decoded_tail_tokens =  create_tokenizer().decode(t_tokens["input_ids"])
        print(decoded_tail_tokens)
        print("------------ head decoded -----------------")
        decoded_head_tokens =  create_tokenizer().decode(h_tokens["input_ids"])
        print(decoded_head_tokens)
        print("---------------------------------------") """
                 
        return {'hr_token_ids': hr_tokens["input_ids"],
                'hr_token_type_ids': hr_tokens["token_type_ids"],
                'tail_token_ids': t_tokens["input_ids"],
                'tail_token_type_ids': t_tokens["token_type_ids"],
                'head_token_ids': h_tokens["input_ids"],
                'head_token_type_ids': h_tokens["token_type_ids"],
                'obj': self}
        

# I still need to check this, but I seems that they only build a neighborhood graph 
# from the training set and also use that for adding neighbors during validation
def add_neighbor_names(head_id, tail_id):
    global neigborhood_graph
    if neigborhood_graph is None:
        neigborhood_graph = build_neighborhood_graph()
    neighbor_ids = neigborhood_graph.get_neighbors(head_id)
    
    if tail_id in neighbor_ids:
        neighbor_ids.remove(tail_id)

    global entities
    if not entities:
        #print("---------------")
        #print(os.path.join("data", args.task, "entities.json"))
        load_entities(os.path.join("data", args.task, "entities.json"))

    neighbor_names = [entities[entity_id].get("entity", "") for entity_id in neighbor_ids]
    return " ".join(neighbor_names)


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path, data_points=None) -> None:
        super().__init__()
        
        self.path = path 
        assert os.path.exists(self.path) or data_points, "Path is invalid: {}".format(path)
        assert path.endswith(".json") or data_points, "Path has wrong formattig. JSON format expected"

        if data_points is None:
            self.data_points = []
            self.data_points = load_data(self.path)
          
        else:
            self.data_points = data_points
            
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, index) -> DataPoint:
        return self.data_points[index].encode_to_dict()
    

def load_data(path: str, add_forward_triplet: bool = True, add_backward_triplet: bool = True) -> List[DataPoint]:
        global entities
        if not entities:
            load_entities(os.path.join(os.path.dirname(args.train_path), "entities.json"))
        
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        
        #logger.info("Load {} datapoints from {}".format(len(data), path))
        
            
        """ if inverse_triples:
            logger.info("Adding inverse triples") """
            
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
                
        #logger.info("Created dataset with {} datapoints".format(len(datapoints)))
            
        return datapoints
    
def collate_fn(batch: List[dict]) -> dict:

    global tokenizer
    if tokenizer is None:
        tokenizer = create_tokenizer()
        
    
    hr_token_ids, hr_mask = batch_token_ids_and_mask(
        [torch.LongTensor(datapoint["hr_token_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id)
    
    tail_token_ids, tail_mask = batch_token_ids_and_mask(
        [torch.LongTensor(datapoint["tail_token_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id)
    
    head_token_ids, head_mask = batch_token_ids_and_mask(
        [torch.LongTensor(datapoint["head_token_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id)
    
    hr_token_type_ids = batch_token_ids_and_mask (
        [torch.LongTensor(datapoint["hr_token_type_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id, create_mask = False)
    
    tail_token_type_ids = batch_token_ids_and_mask (
        [torch.LongTensor(datapoint["tail_token_type_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id, create_mask = False)
    
    head_token_type_ids = batch_token_ids_and_mask (
        [torch.LongTensor(datapoint["head_token_type_ids"]) for datapoint in batch],
        pad_token_id = tokenizer.pad_token_id, create_mask = False)
    
    batch_datapoints = [datapoint["obj"] for datapoint in batch]
    
    return {"batched_hr_token_ids" : hr_token_ids,
            "batched_hr_mask" : hr_mask,
            "batched_hr_token_type_ids" : hr_token_type_ids,
            "batched_tail_token_ids" : tail_token_ids,
            "batched_tail_mask" : tail_mask,
            "batched_tail_token_type_ids" : tail_token_type_ids,
            "batched_head_token_ids" : head_token_ids,
            "batched_head_mask" : head_mask,
            "batched_head_token_type_ids" : head_token_type_ids,
            "batched_datapoints": batch_datapoints,
            "triplet_mask" : construct_triplet_mask(rows=batch_datapoints) if not args.is_test else None,
            "self_neg_mask" : construct_self_negative_mask(batch_datapoints) if not args.is_test else None,}


def batch_token_ids_and_mask(data_batch_tensor, pad_token_id=0, create_mask=True):

    max_length = max([item.size(0) for item in data_batch_tensor])
    num_samples =len(data_batch_tensor)
    batch = torch.LongTensor(num_samples, max_length).fill_(pad_token_id)
    if create_mask:
        mask = torch.ByteTensor(num_samples, max_length).fill_(0)
    for i, tensor in enumerate(data_batch_tensor):
        batch[i, :len(tensor)].copy_(tensor)
        if create_mask:
            mask[i, :len(tensor)].fill_(1)
    
    if create_mask:
        return batch, mask
    else:
        return batch   
    

training_triples_class = None
entity_dict = None

""" def construct_triplet_mask(rows: List[DataPoint], cols: List[DataPoint] = None) -> torch.tensor:
    
    global training_triples_class
    if training_triples_class is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "train.json"))
        training_triples_class = TrainingTripels([file_path])

    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "entities.json"))
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

    return triplet_mask """

def get_entity_dict():
    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "entities.json"))
        entity_dict = EntityDict(file_path)
    return entity_dict

def get_train_triplet_dict():
    global training_triples_class
    if training_triples_class is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "train.txt.json"))
        training_triples_class = TrainingTripels([file_path])
    return training_triples_class

def construct_triplet_mask(rows: List, cols: List = None) -> torch.tensor:
    entity_dict = get_entity_dict()
    training_triples_class = get_train_triplet_dict()
    positive_on_diagonal = cols is None
    num_row = len(rows)
    cols = rows if cols is None else cols
    num_col = len(cols)

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in rows])
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in cols])
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = rows[i].head_id, rows[i].relation
        neighbor_ids = training_triples_class.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = cols[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return ~triplet_mask


def construct_self_negative_mask(datapoints: List[DataPoint]) -> torch.tensor:
    self_mask = torch.zeros(len(datapoints))
    for i, item in enumerate(datapoints):
        neighbors = training_triples_class.get_neighbors(item.get_head_id(), item.get_relation())
        if item.get_head_id() in neighbors:
            self_mask[i] = 1
    return self_mask.bool()