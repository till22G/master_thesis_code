import json
import os

def load_data(path: str):

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

task = 'wiki5m_ind'

train_path = os.path.join('data', task, 'train.txt.json')
test_path = os.path.join('data', task, 'test.txt.json')
valid_path = os.path.join('data', task, 'valid.txt.json' )

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
for triple in test_triples:
    if triple['head_id'] in inductive_entities:
        count += 1
        inductive_triples.append(triple)
        continue
    if triple["tail_id"] in inductive_entities:
        count += 1
        inductive_triples.append(triple)

print('This does effect {} out of the {} test triples. That is {:.4f} %'.format(count, len(test_triples), (count/len(test_triples)*100)))