# load data
import json
import numpy as np


def _load_fbk15_237(path):
    triples = []
    
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                triples.append(line.strip().split('\t'))
                
        print("FBK15_237 data was loaded successfully")
        
    except:
        print("FBK15_237 data could not be loaded successfully")
                              
    return triples
        

def _load_fbk15_237_mid2names(path):
    entity_names = {}
    
    try:
        with open(path, "r",  encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                code, name = line.strip().split('\t')
                name = name.replace("_", " ")
                entity_names[code] = name
        
        print("FBK15_237 names were loaded successfully")
        
    except:      
        print("FBK15_237 names could not be loaded successfully")
        
    return entity_names

def _load_fbk15_237_mid2descriptions(path):
    entity_descriptions = {}
    
    try:
        with open (path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                code, description = line.strip().split('\t')
                description = description.replace("\\", "")
                entity_descriptions[code] = description
                
    except:
        print("FBK15_237_mid2descriptions could not be loaded successfully")

    return entity_descriptions


def fbk15_237_to_json(triples, entity_names):
    entries = []
    
    for triple in triples:
        entry = {}
        entry["head_id"] = triple[0]
        entry["head"] = entity_names[triple[0]]
        entry["realtion"] = triple[1]
        entry["tail_id"] = entity_names[triple[2]]
        entry["tail"] = triple[2]

        entries.append(entry)
        
    filename = "myfile.json"
    try:
        print("Saving file ...") 
        with open(filename, "w") as out_file:
            json.dump(entries, out_file, indent = 6)
            
        print("Data safed to {}".format(filename))
            
    except:
        print("Data could not be saved as .json")
        
        
def _save_FBK15_237_entities_to_json(entity_names, entity_descriptions):
    entries = []
    count = 0
    
    for key in entity_names:
        entry = {}
        entry["entity_id"] = key
        entry["entity"] = entity_names[key]
        if key in entity_descriptions:
            entry["entity_desc"] = entity_descriptions[key]
        else:
            count += 1
            print("Keys not found: {}".format(count))
        
        entries.append(entry)
        
    filename = "eneties.json"
    try:
        print("Saving file ...") 
        with open(filename, "w") as out_file:
            json.dump(entries, out_file, indent = 6)
            
        print("Data saved to {}".format(filename))
            
    except:
        pass
    
            
def main():
    
    print("Start pre-processing")

    triples = _load_fbk15_237("data/FB15K-237/train.txt")
    entity_names = _load_fbk15_237_mid2names("data/FB15K-237/FB15K_mid2name.txt")
    fbk15_237_to_json(triples, entity_names)
    
    entity_descriptions = _load_fbk15_237_mid2descriptions("data/FB15K-237/FB15k_mid2description.txt")
    _save_FBK15_237_entities_to_json(entity_names, entity_descriptions)
        
    print("Finished preprocessing")


if __name__ == "__main__":
    main()

