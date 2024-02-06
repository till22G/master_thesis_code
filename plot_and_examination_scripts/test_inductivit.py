import json
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="test_inductivity")
parser.add_argument("--task", default="WN18RR", type=str)
args = parser.parse_args()

# switch to script dir so paths work
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

task = args.task

print(f"Running task: {task}")

def load_data(path: str):

    print(f"Loading {path}")

    data = json.load(open(path, 'r', encoding='utf-8'))
    entities = set()
    triples = []
    for triple in data:
        head_id = triple.get('head_id', None)
        tail_id = triple.get('tail_id', None)
        entities.add(head_id)
        entities.add(tail_id)
        triples.append(triple)
    return entities, triples

train_path = os.path.join('..', 'rebuild_SimKGC', 'data', task, 'train.json')
test_path = os.path.join('..', 'rebuild_SimKGC', 'data', task, 'test.json')
valid_path = os.path.join('..', 'rebuild_SimKGC','data', task, 'valid.json' )

train_entities, train_triples = load_data(train_path)
test_entities, test_triples = load_data(test_path)
valid_entities, valid_triples = load_data(valid_path)

print('Number of entities in the training set: {}'.format(len(train_entities)))
print('Number of entities in the test set: {}'.format(len(test_entities)))
print('Number of entities in the valid set: {}'.format(len(valid_entities)))

inductive_entities = set()
for entity in test_entities:
    if entity not in train_entities:
        inductive_entities.add(entity)

print('Number of entities in the test set not contained in the training set: {}'.format(len(inductive_entities)))

count = 0
inductive_triples = []
for triple in tqdm(test_triples):
    if triple['head_id'] in inductive_entities:
        count += 1
        inductive_triples.append(triple)
        continue
    if triple["tail_id"] in inductive_entities:
        count += 1
        inductive_triples.append(triple)

print('This effects {} out of the {} test triples. That is {:.4f} %'.format(count, len(test_triples), (count/len(test_triples)*100)))