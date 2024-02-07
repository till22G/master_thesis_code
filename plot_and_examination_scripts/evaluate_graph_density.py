from triplet import EntityDict
import os
import json
from typing import List
from collections import deque, OrderedDict
from dataclasses import dataclass
import time 
import numpy as np
from logger_config import logger
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics 
from transformers import AutoTokenizer
from typing import Optional
from triplet import EntityDict

task = "wiki5m_trans"
max_num_desc_tokens = 50
max_context_size = 10000000
use_relations = True


tokenizer = None
entity_dict = None
training_triples = None
hop_1_graph_new = None

def get_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.join("data", task))
    return entity_dict

def get_training_triples():
    global training_triples
    if not training_triples:
        training_triples = load_data(path=os.path.join("data", task, "train.txt.json"), add_backward_triplet=True)
    return training_triples 

def get_hop_1_graph_new():
    global hop_1_graph_new
    if not hop_1_graph_new:
        hop_1_graph_new = Hop1IndexNew(train_path=os.path.join("data", task, "train.txt.json"),
                                       entity_dict=get_entity_dict(),
                                       max_context_size=max_context_size)
    return hop_1_graph_new
    

@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    

def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(False))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def triples_to_np(triples: List[Example]) -> np.ndarray:
    np_triples = np.empty((len(triples), 3), dtype=object)
    
    for i, triple in enumerate(triples):
        head_inx = entity_dict.entity_to_idx(triple.head_id)
        np_triples[i] = [head_inx, triple.relation, triple.tail]

    return np_triples


class Hop1IndexNew:
    def __init__(self, train_path, entity_dict, key_col=0, max_context_size=10, shuffle=False):
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

def reverse_triplet(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head']
    }

def calculate_avg(token_histogram_data, neighbor_histogram_data):

    token_med = np.median(token_histogram_data)
    token_avg = np.average(token_histogram_data)
    neighbor_med = np.median(neighbor_histogram_data)
    neighbor_avg = np.average(neighbor_histogram_data)


    token_output_dict = {"dataset" : task,
                         "median" : token_med,
                         "average" : token_avg}
    
    neighbor_output_dict = {"dataset" : task,
                         "median" : neighbor_med,
                         "average" : neighbor_avg}
    
    token_file_path = os.path.join('..', 'plots', 'reports', f'report_token_len_list_{task}_max_desc_tokens_{max_num_desc_tokens}_context_size{max_context_size}_with_relations.json')
    neighbor_file_path = os.path.join('..', 'plots', 'reports', f'report_num_neighbors_{task}_context_size{max_context_size}.json')

    with open(token_file_path, "w") as json_file:
        json.dump(token_output_dict, json_file)

    with open(neighbor_file_path, "w") as json_file:
        json.dump(neighbor_output_dict, json_file)

    print(f"median for token length: {token_med}")
    print(f"Avg values for token length: {token_avg}")
    print(f"median for number of neighbors: {neighbor_med}")
    print(f"Avg values for number of neighbors: {neighbor_avg}")
    
    return

def calculate_average_num_neighbors_with_cutoff(histogram, max_num_neighbors):
    neighbor_count = 0
    triples_count = 0
    for key in sorted(histogram.keys()):
        num_triples = histogram[key]
        num_neighbors = key
        if key > max_num_neighbors:
            num_neighbors = max_num_neighbors
        neighbor_count += num_triples * num_neighbors
        triples_count += num_triples

    return neighbor_count / triples_count


def build_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        logger.info('Build tokenizer from {}'.format('distilbert-base-uncased'))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer()
    return tokenizer


def _parse_entity_name(entity: str) -> str:
    if task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _build_context_string(head_id: str, relation: str, tail_id: str, use_desc: bool, entity_dict, hop_1_graph_new):
    context_string = ""
    if head_id == "":
        return ""
    head_idx = entity_dict.entity_to_idx(head_id)
    context = hop_1_graph_new.get(head_idx)
    sep_token = get_tokenizer().sep_token

    for neighbor in context:
        n_relation, n_tail_id = neighbor
        if n_tail_id == tail_id:
            continue
        n_tail_text = _parse_entity_name(entity_dict.get_entity_by_id(n_tail_id).entity)
        ## I might need to shorten the description text
        if use_desc:
            #n_tail_text = _concat_name_desc(n_tail_text, entity_dict.get_entity_by_id(n_tail_id).entity_desc)
            pass
        head_name = _parse_entity_name(entity_dict.get_entity_by_id(head_id).entity)
        if use_relations:
            context_string += f", {n_relation} {n_tail_text}"    
        else:
            context_string += f", {n_tail_text}"
        if context_string == ", ":
            return ""
    
    return f"{context_string}"


def _tokenize(text1: str, text2: Optional[str] = None, text_pair: Optional[str] = None ) -> dict:
    tokenizer = get_tokenizer()
    
    text1_encodings = tokenizer.encode(text1, max_length=max_num_desc_tokens, truncation=True, add_special_tokens=False)
    text1 = tokenizer.decode(text1_encodings)

    if text2:
        text = text1 + " : " + text2
    else:
        text = text1

    encodings_total_length = tokenizer(text=text,
                             text_pair=text_pair if text_pair else None,
                             add_special_tokens=True,
                             return_token_type_ids=True,
                             max_length=100000, # set a high value , otherwise the default is model_max_length
                             truncation=True)


    encodings = tokenizer(text=text,
                          text_pair=text_pair if text_pair else None,
                          add_special_tokens=True,
                          max_length=tokenizer.model_max_length,
                          return_token_type_ids=True,
                          truncation=True
                        )

    return encodings, encodings_total_length


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity

def calculate_num_tokens():
    
    if use_relations:
        data_path_1 = os.path.join('..', 'plots', 'plot_data', f'token_len_list_{task}_max_desc_tokens_{max_num_desc_tokens}_context_size{max_context_size}_with_relations.npy')
        data_path_2 = os.path.join('..', 'plots', 'plot_data', f'token_total_len_list_{task}_max_desc_tokens_{max_num_desc_tokens}_context_size{max_context_size}_with_relations.npy')
        
    else:
        data_path_1 = os.path.join('..', 'plots', 'plot_data', f'token_len_list_{task}_max_desc_tokens_{max_num_desc_tokens}_context_size{max_context_size}_without_relations.npy')
        data_path_2 = os.path.join('..', 'plots', 'plot_data', f'token_total_len_list_{task}_max_desc_tokens_{max_num_desc_tokens}_context_size{max_context_size}_without_relations.npy')
        
    data_path_3 = os.path.join('..', 'plots', 'plot_data', f'number_of_neighbors_{task}_context_size{max_context_size}.npy')

    if os.path.exists(data_path_1) and os.path.exists(data_path_2) and os.path.exists(data_path_3):
        print(f"Loading token data from: {data_path_1}")
        token_len_list = np.load(data_path_1)
        print(f"Loading total token data form {data_path_2}")
        token_total_len_list = np.load(data_path_2)
        print(f"Loading number of neighbors data form {data_path_3}")
        number_of_neighbors = np.load(data_path_3)

    else:
        if not os.path.exists(data_path_1):
            print(f"No exisiting data found for: {data_path_1}")
        if not os.path.exists(data_path_2):
            print(f"No exisiting data found for: {data_path_2}")
        if not os.path.exists(data_path_3):
            print(f"No exisiting data found for: {data_path_3}")
        
        entity_dict = get_entity_dict()
        training_triples = get_training_triples()
        hop_1_graph_new = get_hop_1_graph_new()

        token_len_list = []
        token_total_len_list = []
        number_of_neighbors = []
        for triple in tqdm(training_triples):
            head_id = triple.head_id
            tail_id = triple.tail_id
            head_desc = triple.head_desc
            
            head_desc = triple.head_desc
            head_with_desc = _concat_name_desc(_parse_entity_name(head_id), head_desc)
            context_string = _build_context_string(head_id=head_id, relation=triple.relation, tail_id=tail_id, use_desc=False, entity_dict=entity_dict, hop_1_graph_new=hop_1_graph_new)
            tokens, tokens_total_len = _tokenize(text1=head_with_desc, text2=context_string)

            token_total_len_list.append(len(tokens_total_len['input_ids']))
            token_len_list.append(len(tokens['input_ids']))

            # get number of neighbors
            neigbors = hop_1_graph_new.get_neighbors(head_id)
            number_of_neighbors.append(len(neigbors))
        
        np.save(data_path_1, token_len_list)
        np.save(data_path_2, token_total_len_list)
        np.save(data_path_3, number_of_neighbors)

    calculate_avg(token_len_list, number_of_neighbors)

    return token_len_list, token_total_len_list, number_of_neighbors


def print_hist_of_num_tokens(data, data_total, bin_size=1):
    
    plt.hist(data, bins=range(min(data), max(data) + bin_size, bin_size), edgecolor='black')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(f'Histogram number of tokens for entity descriptions {task}')
    if use_relations:
        save_path = f"../plots/plots/histogram_num_tokens_{task}_desc_{max_num_desc_tokens}_context_size_{max_context_size}_with_relations"
    else:
        save_path = f"../plots/plots/histogram_num_tokens_{task}_desc_{max_num_desc_tokens}_context_size_{max_context_size}_without_relations"
    plt.savefig(save_path, dpi=300)
    plt.clf()

    # calculate 95th percentile
    #percentile_95_value = np.percentile(data_total, 95)

    plt.hist(data_total, bins=range(min(data_total), max(data_total) + bin_size, bin_size), edgecolor='black')
    #plt.axvline(x=percentile_95_value, color='green', linestyle='--', label='95th Percentile')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(f'Histogram number of tokens for entity descriptions {task}')
    if use_relations:
        save_path = f"../plots/plots/histogram_num_total_tokens_{task}_desc_{max_num_desc_tokens}_context_size_{max_context_size}_with_relations"
    else:
        save_path = f"../plots/plots/histogram_num_total_tokens_{task}_desc_{max_num_desc_tokens}_context_size_{max_context_size}_without_relations"
    plt.savefig(save_path, dpi=300)
    plt.clf()


def print_hist_num_neighbors(data, bin_size=1):
    plt.hist(data, bins=range(min(data), max(data) + bin_size, bin_size), edgecolor='black')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Frequency')
    plt.title(f'Histogram number of neighbors for {task}')

    save_path = f"../plots/plots/histogram_num_of_neighbors_{task}_{max_context_size}"
    plt.savefig(save_path, dpi=300)
    plt.clf()

    
token_len_list, token_total_len_list, number_of_neighbors = calculate_num_tokens()

print_hist_of_num_tokens(token_len_list, token_total_len_list, 1)
print_hist_num_neighbors(number_of_neighbors)