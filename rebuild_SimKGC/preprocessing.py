# load data
import json

train_entries = []

train_tripels = []
with open("data/FB15K-237/train.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:
        train_tripels.append(line.strip().split('\t'))
        
        
train_entity_names = {}
with open("data/FB15K-237/FB15K_mid2name.txt", "r",  encoding="utf-8") as file:
    lines = file.readlines()
    c = 0
    for line in lines:
        code, name = line.strip().split('\t')
        name = name.replace("_", " ")
        train_entity_names[code] = name

for triple in train_tripels:
    entry = {}
    entry["head_id"] = triple[0]
    entry["head"] = train_entity_names[triple[0]]
    entry["realtion"] = triple[1]
    entry["tail_id"] = train_entity_names[triple[2]]
    entry["tail"] = triple[2]

    train_entries.append(entry)
    
    
  
# the json file where the output must be stored
with open("myfile.json", "w") as out_file:
    json.dump(train_entries, out_file, indent = 6)
    
print("Script done")