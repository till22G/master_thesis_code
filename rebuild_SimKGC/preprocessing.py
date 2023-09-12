# load data
import json


def _load_fbk15_237(path):
    triples = []
    
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                triples.append(line.strip().split('\t'))
    except:
        print("FBK15_237 data could not be loaded")
                
    print("Data successfully loaded")                

    return triples
        
        
def _load_fbk15_237_names(path):
    entity_names = {}
    
    try:
        with open(path, "r",  encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                code, name = line.strip().split('\t')
                name = name.replace("_", " ")
                entity_names[code] = name
    except:
        print("FBK15_237 names could not be loaded")
              
    print("Names successfully loaded")
    return entity_names


def fbk15_237_to_json(triples, entity_names):
    train_entries = []
    
    for triple in triples:
        entry = {}
        entry["head_id"] = triple[0]
        entry["head"] = entity_names[triple[0]]
        entry["realtion"] = triple[1]
        entry["tail_id"] = entity_names[triple[2]]
        entry["tail"] = triple[2]

        train_entries.append(entry)
        
    try:
        print("Saving file ...") 
        with open("myfile.json", "w") as out_file:
            json.dump(train_entries, out_file, indent = 6)
            
    except:
        print("data could not be safed as .json") 
            
            
def main():
    
    print("Start pre-processing")

    triples = _load_fbk15_237("data/FB15K-237/train.txt")
    entity_names = _load_fbk15_237_names("data/FB15K-237/FB15K_mid2name.txt")
    fbk15_237_to_json(triples, entity_names)


if __name__ == "__main__":
    main()

    
print("Script done")