# load data
import json

def _load_fbk15_237(dataset):
    triples = []
    
    try:
        with open("data/FB15K-237/{dataset}.txt".format(dataset=dataset), "r", encoding="utf-8") as file:
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


def fbk15_237_to_json(triples, entity_names, dataset):
    entries = []
    
    for triple in triples:
        entry = {}
        entry["head_id"] = triple[0]
        entry["head"] = entity_names[triple[0]]
        entry["relation"] = _process_relation(triple[1])
        entry["tail_id"] = triple[2] 
        entry["tail"] = entity_names[triple[2]]

        entries.append(entry)
        
    filename = "data/FB15K-237/{dataset}.json".format(dataset=dataset)
    try:
        print("Saving FBK15-237 triples as {} ...".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            
        print("Data saved to {}".format(filename))
            
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
            print("Key not found in entity descriptions: {}".format(key))
        
        entries.append(entry)
        
    print("{} keys do not exist in entity descriptions".format(count))
        
    filename = "data/FB15K-237/entities.json"
    try:
        print("Saving FBK15-237 entity data as {} ...".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            
        print("Entities saved to {}".format(filename))
            
    except:
        print("Entities could not be saved")
    
def _process_relation(relation: str) -> str: 
    return relation.replace("/", " ").replace("./", " ").replace("_", " ")

            
def main():
    
    print("Start pre-processing")
    
    datasets = ["train", "valid", "test"]
    
    for dataset in datasets:
        triples = _load_fbk15_237(dataset)
        entity_names = _load_fbk15_237_mid2names("data/FB15K-237/FB15K_mid2name.txt")
        fbk15_237_to_json(triples, entity_names, dataset)
    
    entity_descriptions = _load_fbk15_237_mid2descriptions("data/FB15K-237/FB15k_mid2description.txt")
    _save_FBK15_237_entities_to_json(entity_names, entity_descriptions)

    print("Finished preprocessing")


if __name__ == "__main__":
    main()

