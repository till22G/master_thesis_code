import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import List, Optional
from transformers import AutoTokenizer
from collections import deque

from logger import logger
from argparser import args

script_dir = os.path.dirname(__file__)
tokenizer: AutoTokenizer = None
entities = {}
neigborhood_graph = None
training_triples_class = None
entity_dict = None

class EntityDict():
    def __init__(self, path: str, inductive_test_path: str = None) -> None:
        self.entities = []
        self.id2entity = {}
        self.entity2idx = {}
        
        self._load_entity_dict(path, inductive_test_path)
    
    def _load_entity_dict(self, path: str, inductive_test_path: str = None):

        if inductive_test_path:
            path = os.path.join(os.path.dirname(inductive_test_path), "entities.json") 
        assert os.path.exists(path), "Path is invalid {}:".format(path)
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
        with open(path, "r", encoding="utf8") as infile:
            data = json.load(infile)
            
        self.entities = data

        if inductive_test_path:
            with open(inductive_test_path, "r", encoding="utf-8") as infile:
                inductive_data = json.load(infile)

            entity_ids = set()
            for triple in inductive_data:
                entity_ids.add(triple['head_id'])
                entity_ids.add(triple['tail_id'])
            self.entities = [entity for entity in self.entities if entity["entity_id"] in entity_ids]

        self.id2entity = {entity["entity_id"]: entity for entity in self.entities}
        self.entity2idx = {entity["entity_id"]: i for i, entity in enumerate(self.entities)}
    
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

         # load triples from all specified datasets
        for path in path_list:
            self._load_training_tripels(path)
        
    def _load_training_tripels(self, path) -> None:
        
        assert os.path.exists(path), "Path is invalid {path}"
        assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
        
        with open(path, "r", encoding="utf-8") as inflile:
            triples = json.load(inflile)
                
        for triple in triples:
            self.training_triples.append(triple)
            if args.use_inverse_triples:
                inverse_triple = {"head_id": triple["tail_id"],
                                  "head" : triple["tail"],
                                  "relation": " ".join(("inverse", triple["relation"])),
                                  "tail_id": triple["head_id"],
                                  "tail" : triple["head"]}
                self.training_triples.append(inverse_triple)
                
        # create mapping from head-relation to tail
        for triple in self.training_triples:
            key = (triple["head_id"], triple["relation"])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(triple["tail_id"])

    def get_neighbors(self, head_id: str, relation: str) -> set:
        return self.hr2tails.get((head_id, relation), set())
            

class NeighborhoodGraph:
    def __init__(self, train_path, entity_dict, key_col=0, max_context_size=args.max_context_size, shuffle=False):
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

    def _sort_context(self, context):
        if args.most_common_first: 
            i = -1
        elif args.least_common_first: 
            i = 1

        unique_relations, counts = np.unique(context[:, 0], return_counts=True)
        sorted_elements = unique_relations[np.argsort(i*counts)]
        element_to_index = {element: index for index, element in enumerate(sorted_elements)}
        sorted_context = context[np.argsort(np.vectorize(element_to_index.get)(context[:, 0]))]
        print("==================")
        print(sorted_context)
        print("==================")
        return sorted_context

    def __getitem__(self, item):
        start = self.key_to_start[item]
        end = self.key_to_end[item]
        context = self.triples[start:end, [1, 2]]
        if self.shuffle:
            context = np.copy(context)
            np.random.shuffle(context)
        if args.most_common_first or args.least_common_first:
            context = self._sort_context(context)
        if end - start > self.max_context_size: 
            context = context[:self.max_context_size]
        print("------------------")
        print(context)
        print("------------------")
        return context
    
    def get(self, item):
        return self[item]

    def get_neighbors(self, id):
        if id == '':
            return None
        entity_dict = build_entity_dict()
        idx = entity_dict.entity_to_idx(id)
        return self[idx]
    
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

    
def build_entity_dict():
    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "entities.json"))
        entity_dict = EntityDict(file_path)
    return entity_dict
    

def build_neighborhood_graph():
    global neigborhood_graph
    if neigborhood_graph == None:
        entity_dict = build_entity_dict()
        neigborhood_graph = NeighborhoodGraph(args.train_path, entity_dict=entity_dict)
    return neigborhood_graph
    
    
def load_entities(path) -> None:
    
    assert os.path.exists(path), "Path is invalid: {}".format(path)
    assert path.endswith(".json"), "Path has wrong formattig. JSON format expected"
    
    with open (path, "r", encoding="utf-8") as infile:
        logger.info("Loading entity descriptions from {}".format(path))
        data = json.load(infile)
            
    for item in data:
        entities[item["entity_id"]] = item
        
    logger.info("{} entity descriptinos loaded".format(len(entities)))
    

def create_tokenizer():
    global tokenizer
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    return tokenizer


def add_neighbor_names(head_id, tail_id):
    neighbors = build_neighborhood_graph().get_neighbors(head_id)
    neighbor_string = ""
    if neighbors is None:
        return ""
    for neighbor in neighbors:
        _ , neighbor_id = neighbor
        if not args.is_test and neighbor_id == tail_id:
            continue
        n_tail_name = entity_dict.get_entity_by_id(neighbor_id)["entity"]
        neighbor_string += f", {n_tail_name}"
    return neighbor_string


def _tokenize(head: str, context: Optional[str] = None, text_pair: Optional[str] = None ) -> dict:
    tokenizer = create_tokenizer()
    # encode head (including descriptions) and truncate the string to the max number of tokens set
    head_encodings = tokenizer.encode(head, max_length=args.max_num_desc_tokens, truncation=True, add_special_tokens=False)
    head = tokenizer.decode(head_encodings) # decode to get the string bet for better string manipulation
    # concatenate of context is passed along
    if context: 
        text = head + " : " + context
    else: 
        text = head

    encodings = tokenizer(text=text,
                          text_pair=text_pair if text_pair else None,
                          add_special_tokens=True,
                          max_length=args.max_number_tokens,
                          # max_length=512, # tokenizer.model_max_length results in an error
                          return_token_type_ids=True,
                          truncation=True)
    return encodings


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def _build_context_string(head_id: str, relation: str, tail_id: str, max_context_size: int, use_context_descriptions: bool):
    context_string = ""
    if head_id == "":
        return ""
    entity_dict = build_entity_dict()
    head_idx = entity_dict.entity_to_idx(head_id)
    context = build_neighborhood_graph().get(head_idx)

    for neighbor in context[:max_context_size]:
        n_relation, neighbor_id = neighbor
        if not args.is_test and neighbor_id == tail_id:
            continue
        n_text = entity_dict.get_entity_by_id(neighbor_id)["entity"]
        if use_context_descriptions:
            n_tail_desc = ' '.join(entity_dict.get_entity_by_id(neighbor_id)["entity_desc"].split()[:40])
            n_text = _concat_name_desc(n_text, n_tail_desc)
        if args.use_context_relation:
            context_string += f", {n_relation} {n_text}"
        else:
            context_string += f", {n_text}"
        if context_string == ", ":
            return ""
    
    return f"{context_string[2:]}"


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
        
        head_desc, tail_desc = self.get_head_desc(), self.get_tail_desc()
        head_context = ''
        tail_context = ''
        if args.use_neighbors:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + add_neighbor_names(head_id=self.get_head_id(), tail_id=self.get_tail_id())
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + add_neighbor_names(head_id=self.get_tail_id(), tail_id=self.get_head_id())

            head_word = _concat_name_desc(self.get_head(), head_desc)
            tail_word = _concat_name_desc(self.get_tail(), tail_desc)

        if args.use_head_context:
            head_word = self.get_head()
            if args.use_descriptions:
                head_word = _concat_name_desc(head_word, head_desc)

            head_context = _build_context_string(self.get_head_id(), self.get_relation(), self.get_tail_id(), 
                                                 max_context_size = args.max_context_size,
                                                 use_context_descriptions = args.use_context_descriptions)
            
            if not args.use_tail_context:
                tail_word = self.get_tail()
                if args.use_descriptions:
                    tail_word = _concat_name_desc(tail_word, tail_desc)
        
        if args.use_tail_context:
            tail_word = self.get_tail()
            if args.use_descriptions:
                tail_word = _concat_name_desc(tail_word, tail_desc)

            tail_context = _build_context_string(self.get_tail_id(), self.get_relation(), self.get_head_id(),
                                                 max_context_size = args.max_context_size,
                                                 use_context_descriptions = args.use_context_descriptions)
        
            
            if not args.use_head_context:
                head_word = self.get_head()
                if args.use_descriptions:
                    head_word = _concat_name_desc(head_word, head_desc)    


        if not (args.use_neighbors or args.use_head_context or args.use_tail_context):
            head_word = self.get_head()
            if args.use_descriptions:
                head_word = _concat_name_desc(head_word, head_desc)

            tail_word = self.get_tail()
            if args.use_descriptions:
                tail_word = _concat_name_desc(tail_word, tail_desc)

         
        text_pair = self.relation
        hr_tokens = _tokenize(head=head_word,
                              context=head_context,
                              text_pair=text_pair)

        h_tokens = _tokenize(head=head_word)

        
        t_tokens = _tokenize(head=tail_word,
                             context=tail_context,)
        
        """ print("---------------- Head-relation decoded ----------------")
        decoded_hr_tokens =  create_tokenizer().decode(hr_tokens["input_ids"])
        print(decoded_hr_tokens)

        #print("original:")
        #print(head_word)
        #print(head_context)
        #print("-"*100)

        print("---------------- Tail decoded ----------------")
        decoded_tail_tokens =  create_tokenizer().decode(t_tokens["input_ids"])
        print(decoded_tail_tokens)

        #print("original:")
        #print(tail_word)
        #print(tail_context)
        #print("-"*100)

        print("---------------- Head decoded ----------------")
        decoded_h_tokens =  create_tokenizer().decode(h_tokens["input_ids"])
        print(decoded_h_tokens)
        print("---------------------------------------") """

        return {'hr_token_ids': hr_tokens["input_ids"],
                'hr_token_type_ids': hr_tokens["token_type_ids"],
                'tail_token_ids': t_tokens["input_ids"],
                'tail_token_type_ids': t_tokens["token_type_ids"],
                'head_token_ids': h_tokens["input_ids"],
                'head_token_type_ids': h_tokens["token_type_ids"],
                'obj': self}
        
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
    
def collate_fn(batch: List[dict]) -> dict:

    global tokenizer
    if tokenizer is None:
        tokenizer = create_tokenizer()

    hr_token_ids   = [torch.LongTensor(datapoint["hr_token_ids"]) for datapoint in batch]
    hr_token_type_ids  = [torch.LongTensor(datapoint["hr_token_type_ids"]) for datapoint in batch]
    tail_token_ids = [torch.LongTensor(datapoint["tail_token_ids"]) for datapoint in batch]
    tail_token_type_ids = [torch.LongTensor(datapoint["tail_token_type_ids"]) for datapoint in batch]
    head_token_ids = [torch.LongTensor(datapoint["head_token_ids"]) for datapoint in batch]
    head_token_type_ids = [torch.LongTensor(datapoint["head_token_type_ids"]) for datapoint in batch]

    batched_hr_token_ids, batched_hr_mask = batch_token_ids_and_mask(hr_token_ids, pad_token_id = tokenizer.pad_token_id)
    batched_hr_token_type_ids = batch_token_ids_and_mask (hr_token_type_ids, pad_token_id = tokenizer.pad_token_id, create_mask = False)
    batched_tail_token_ids, batched_tail_mask = batch_token_ids_and_mask(tail_token_ids, pad_token_id = tokenizer.pad_token_id)
    batched_tail_token_type_ids = batch_token_ids_and_mask (tail_token_type_ids, pad_token_id = tokenizer.pad_token_id, create_mask = False)
    batched_head_token_ids, batched_head_mask = batch_token_ids_and_mask(head_token_ids, pad_token_id = tokenizer.pad_token_id)
    batched_head_token_type_ids = batch_token_ids_and_mask (head_token_type_ids, pad_token_id = tokenizer.pad_token_id, create_mask = False)

    batched_datapoints = [datapoint["obj"] for datapoint in batch]

    return {"batched_hr_token_ids" : batched_hr_token_ids,
            "batched_hr_mask" : batched_hr_mask,
            "batched_hr_token_type_ids" : batched_hr_token_type_ids,
            "batched_tail_token_ids" : batched_tail_token_ids,
            "batched_tail_mask" : batched_tail_mask,
            "batched_tail_token_type_ids" : batched_tail_token_type_ids,
            "batched_head_token_ids" : batched_head_token_ids,
            "batched_head_mask" : batched_head_mask,
            "batched_head_token_type_ids" : batched_head_token_type_ids,
            "batched_datapoints": batched_datapoints,
            "triplet_mask" : construct_triplet_mask(rows=batched_datapoints) if not args.is_test else None,
            "self_neg_mask" : construct_self_negative_mask(batched_datapoints) if not args.is_test else None,}


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

def construct_triplet_mask(rows: List[DataPoint], cols: List[DataPoint] = None) -> torch.tensor:
    
    global training_triples_class
    if training_triples_class is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "train.json"))
        training_triples_class = TrainingTripels([file_path])

    global entity_dict
    if entity_dict is None:
        file_path = os.path.join(script_dir, os.path.join("data", args.task, "entities.json"))
        entity_dict = EntityDict(file_path)
        
    num_rows = len(rows)
    cols_none = cols is None
    cols = rows if cols_none else cols
    num_cols = num_rows if cols_none else len(cols)
            
    tail_ids_rows = torch.LongTensor([entity_dict.entity_to_idx(datapoint.get_tail_id()) for datapoint in rows])
    tail_ids_cols = tail_ids_rows if cols_none \
        else torch.LongTensor([entity_dict.entity_to_idx(datapoint.get_tail_id()) for datapoint in cols])
    
    triplet_mask = tail_ids_rows.unsqueeze(1) == tail_ids_cols.unsqueeze(0)
    if cols_none:
        triplet_mask.fill_diagonal_(False)
 
    # mask out true triples
    for i in range(num_rows):
        head, relation = rows[i].get_head_id(), rows[i].get_relation()
        neighbors = training_triples_class.get_neighbors(head, relation)
        if len(neighbors) <= 1: continue
        for j in range(num_cols):
            if i == j and cols_none: continue 
            if cols[j].tail_id in neighbors:
                triplet_mask[i][j] = True

    return triplet_mask

def construct_self_negative_mask(datapoints: List[DataPoint]) -> torch.tensor:
    self_mask = torch.zeros(len(datapoints))
    for i, item in enumerate(datapoints):
        neighbors = training_triples_class.get_neighbors(item.get_head_id(), item.get_relation())
        if item.get_head_id() in neighbors:
            self_mask[i] = 1
    return self_mask.bool()