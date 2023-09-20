# load data
import json

def _load_fbk15_237(dataset_path: str, dataset: str) -> list:
    triples = []
    
    try:
        with open("{dataset_path}{dataset}.txt".format(dataset_path=dataset_path, dataset=dataset), "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                triples.append(line.strip().split('\t'))
                
        print("FBK15_237 data was loaded successfully")
        
    except Exception as e:
        print("FBK15_237 data could not be loaded successfully")
        print(e)
                              
    return triples
        

def _load_fbk15_237_mid2names(path: str) -> dict:
    entity_names = {}
    
    try:
        with open(path, "r",  encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                code, name = line.strip().split('\t')
                name = name.replace("_", " ")
                entity_names[code] = name
        
        print("{} names from /FB15k_mid2description.txt were loaded successfully".format(len(entity_names)))
        
    except Exception as e:      
        print("FBK15_237 names could not be loaded successfully")
        print(e)
        
    return entity_names

def _load_fbk15_237_mid2descriptions(path: str) -> dict:
    entity_descriptions = {}
    
    try:
        with open (path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                code, description = line.strip().split('\t')
                description = description.replace("\\", "")
                entity_descriptions[code] = " ".join(description.split()[:50]) # truncate descriptions at 50 tokens
                
    except Exception as e:
        print("FBK15_237_mid2descriptions could not be loaded successfully")
        print(e)

    return entity_descriptions


def fbk15_237_to_json(triples: list, entity_names: dict, dataset_path: str, dataset: str) -> None:
    entries = []
    
    rel_id_to_surface_form = _normalize_relations(triples)
    _check_duplicates(rel_id_to_surface_form)
    
    for triple in triples:
        entry = {}
        entry["head_id"] = triple[0]
        entry["head"] = entity_names[triple[0]]
        entry["relation"] = rel_id_to_surface_form[triple[1]]
        entry["tail_id"] = triple[2] 
        entry["tail"] = entity_names[triple[2]]

        entries.append(entry)
        
    filename = "{dataset_path}{dataset}.json".format(dataset_path=dataset_path, dataset=dataset)
    try:
        print("Saving FBK15-237 triples as {} ...".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            
        print("Data saved to {}".format(filename))
            
    except Exception as e:
        print("Data could not be saved as .json")
        print(e)
        
        
def _save_FBK15_237_entities_to_json(entity_names: dict, entity_descriptions: dict, dataset_path: str) -> None:
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
        
    filename = "{dataset_path}preprocessed_entities.json".format(dataset_path=dataset_path)
    try:
        print("Saving FBK15-237 entity data as {} ...".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            
        print("Entities saved to {}".format(filename))
            
    except Exception as e:
        print("Entities could not be saved")
        print(e)
    
def _rel_to_surface_form(relation: str) -> str: 
    return relation.replace("/", " ").replace("./", " ").replace("_", " ")


def _normalize_relations(triples: list) -> dict:
    rel_id_to_surface_form = {}
    for item in triples:
        rel_id_to_surface_form[item[1]] = _rel_to_surface_form(item[1])

    return rel_id_to_surface_form 
    

def _check_duplicates(rel_id_to_surface_form: dict) -> None:
    
    """ tmp = rel_id_to_surface_form.copy()
    tuple = tmp.popitem()
    rel_id_to_surface_form["test_key"] = tuple[1] """

    surface_form_to_rel_id = {}
    for key, item in rel_id_to_surface_form.items():
        if item is None:
            continue
        surface_form_to_rel_id.setdefault(item, set()).add(key) 
    result = [key for key, values in surface_form_to_rel_id.items() if len(values) > 1]
    if (len(result) == 0):
        print("No duplicates in surface forms found")
    else:
        print("Attention !!! Some relations normalize to the same surface form")
        print(result)


def main():
    
    print("Start pre-processing")
    
    datasets = ["train", "valid", "test"]
    
    dataset_path = "data/FB15K237/"
    
    # list containing a dictionary of each triple in the data. The dictionary containes
    # the head_id, the head_name, the normalized relation, the tail_id and the tail name
    all_triples = []
    
    # load entity descriptions as dictionary with the the code as key and the descriptions as value 
    entity_descriptions = _load_fbk15_237_mid2descriptions("{dataset_path}FB15k_mid2description.txt".format(dataset_path=dataset_path))
    entity_names = _load_fbk15_237_mid2names("{dataset_path}FB15K_mid2name.txt".format(dataset_path=dataset_path))
    
    for dataset in datasets:
        triples = _load_fbk15_237(dataset_path, dataset)
        fbk15_237_to_json(triples, entity_names, dataset_path, dataset)
    
    
    
    _save_FBK15_237_entities_to_json(entity_names, entity_descriptions, dataset_path)

    print("Finished preprocessing")


if __name__ == "__main__":
    main()

