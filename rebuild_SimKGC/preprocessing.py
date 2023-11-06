import json
import traceback
import os

from argparser import args
from logger import logger

def _load_fbk15_237_triples(dataset_path: str, dataset: str) -> list:
    triples = []
    path = os.path.join(dataset_path, "{}.txt".format(dataset))
    
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                triples.append(line.strip().split('\t'))
                
        logger.info("{} FBK15_237 triples loaded successfully".format(len(triples)))
        
    except Exception as e:
        global error_count
        error_count += 1
        logger.error("FBK15_237 data could not be loaded successfully")
        logger.error(traceback.print_exc())
             
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
        
        logger.info("{} names from FB15k_mid2names loaded successfully".format(len(entity_names)))
        
    except Exception as e:
        global error_count
        error_count += 1
        logger.error("FBK15_237_mid2names could not be loaded")
        logger.error(traceback.print_exc())

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
        logger.info("FBK15_237_mid2descriptions loaded successfully")
                
    except:
        global error_count
        error_count += 1
        logger.error("FBK15_237_mid2descriptions could not be loaded successfully")
        logger.error(traceback.print_exc())

    return entity_descriptions


def fbk15_237_to_json(triples: list, entity_names: dict, dataset_path: str, dataset: str) -> None:
    entries = []
    
    rel_id_to_surface_form = _normalize_relations(triples, save_relations=True, data_dir=dataset_path)
    _check_duplicates(rel_id_to_surface_form)
    
    for triple in triples:
        entry = {}
        entry["head_id"] = triple[0]
        entry["head"] = entity_names[triple[0]]
        entry["relation"] = rel_id_to_surface_form[triple[1]]
        entry["tail_id"] = triple[2] 
        entry["tail"] = entity_names[triple[2]]

        entries.append(entry)
        
    filename = os.path.join(dataset_path, "{}.json".format(dataset))
    try:
        logger.info("Saving FBK15-237 triples as {}".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            
        logger.info("Data saved to {}".format(filename))
            
    except Exception as e:
        global error_count
        error_count += 1 
        print("Data could not be saved as .json")
        print(e)
        
                
def _save_entities_to_json(all_triples: list, entity_names: dict, entity_descriptions: dict, dataset_path: str) -> None:
   
    count = 0
    entities = {}
    
    # save all entities that are contained in the training, validation and test data
    for triple in all_triples:
        head_id = triple[0]
        if head_id not in entities:
            entity_name = entity_names.get(head_id, None)
            if head_id in entity_descriptions:
                entity_description = entity_descriptions[head_id]
            else:
                count += 1
                logger.warning("Entity ID not found in entity descriptions: {}".format(head_id))
                
            entities[head_id] = {"entity_id" : head_id,
                                 "entity": entity_name,
                                 "entity_desc": entity_description}
                     
        tail_id = triple[2]
        if tail_id not in entities:
            entity_name = entity_names.get(tail_id, None)
            if tail_id in entity_descriptions:
                entity_description = entity_descriptions[tail_id]
            else:
                count += 1
                logger.warning("Entity ID not found in entity descriptions: {}".format(tail_id))
                
            entities[tail_id] = {"entity_id" : tail_id,
                                 "entity": entity_name,
                                 "entity_desc": entity_description}
            
        
    filename = os.path.join(dataset_path, "entities.json")
    try:
        logger.info("Saving entity data as {}".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            #json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            json.dump(list(entities.values()), out_file, indent = 4, ensure_ascii=False)
            
        logger.info("Entities saved to {}".format(filename))
            
    except Exception as e:
        global error_count
        error_count += 1 
        logger.error("Entities could not be saved")
        logger.error(traceback.print_exc())
    
    

# !!! this function is taken from the official SimKGC repository !!!
def _rel_to_surface_form(relation: str) -> str: 
    tokens = relation.replace("/", "/").replace("./", "/").strip().split("/")    
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)
                         if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    relation = relation.replace("_", " ")

    return relation

# !! check whether I am allowed to use this funciton or not!!!


def _normalize_relations(triples: list, save_relations: bool=True, data_dir: str=None) -> dict:
    rel_id_to_surface_form = {}
    for item in triples:    
        rel_id_to_surface_form[item[1]] = _rel_to_surface_form(item[1])
        
    if save_relations:
        filepath = os.path.join(data_dir, "relations.json")
        try:
            with open(filepath, "w", encoding="utf-8") as out_file:
                json.dump(rel_id_to_surface_form, out_file, ensure_ascii=False, indent=4)
                logger.info("Save {} realtions to \"{}\"".format(len(rel_id_to_surface_form), filepath))

        except Exception as e:
            global error_count
            error_count += 1 
            logger.warning("Entities could not be saved")
            logger.warning(traceback.print_exc())
        
    return rel_id_to_surface_form 
    

def _check_duplicates(rel_id_to_surface_form: dict) -> None:

    surface_form_to_rel_id = {}
    for key, item in rel_id_to_surface_form.items():
        if item is None:
            continue
        surface_form_to_rel_id.setdefault(item, set()).add(key) 
    result = [key for key, values in surface_form_to_rel_id.items() if len(values) > 1]
    if (len(result) == 0):
        logger.info("No duplicates in surface forms found")
    else:
        logger.warning("Attention! Some relations normalize to the same surface form")
        logger.warning(result)

################################################################################################
def _load_wn18rr_descriptions(path):
    entity_descriptions = {}
    
    try:
        with open (path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                code, word, description = line.strip().split('\t')
                word = " ".join(word.replace("__", "").split("_")[:-2]).strip()
                entity_descriptions[code] = (code, word, description)
        logger.info("wn18rr definitions loaded successfully")
                
    except:
        global error_count
        error_count += 1
        logger.error("wn18rr definitions could not be loaded successfully")
        logger.error(traceback.print_exc())

    return entity_descriptions


def _load_wn18rr_relations(path: str) -> dict:
    wn18rr_relations = {}

    try:
        with open(path, "r", encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                split_line = line.strip().split("\t")
                wn18rr_relations[split_line[1]] = " ".join(split_line[1].split("_")).strip()
        logger.info("wn18rr.dict loaded successfully ({})".format(args.task))

    except:
        global error_count
        error_count += 1
        logger.error("Wikidata5m_relation.txt could not be loaded successfully")
        logger.error(traceback.print_exc())

    return wn18rr_relations


def _load_wn18rr_triples(dataset_path, dataset):
    triples = []
    path = os.path.join(dataset_path, "{}.txt".format(dataset))
    
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                triples.append(line.strip().split('\t'))
                
        logger.info("{} wn18rr triples loaded successfully".format(len(triples)))
        
    except Exception as e:
        global error_count
        error_count += 1
        logger.error("wn18rr data could not be loaded successfully")
        logger.error(traceback.print_exc())
             
    return triples

def _separate_wr18nn_names_and_description(names_and_descriptions: dict) -> (dict, dict):
    
    entity_names = {}
    entity_descriptions = {}

    for key in names_and_descriptions:
        _, name, description = names_and_descriptions[key]
        entity_names[key] = name
        entity_descriptions[key] = description
    
    return (entity_names, entity_descriptions)



def wn18rr_to_json(triples: list, entity_names: dict, realtion_id_to_surface_form: dict, dataset_path: str, dataset: str) -> None:
    entries = []
    
    for triple in triples:
        entry = {}
        entry["head_id"] = triple[0]
        entry["head"] = entity_names[triple[0]]
        entry["relation"] = realtion_id_to_surface_form[triple[1]]
        entry["tail_id"] = triple[2] 
        entry["tail"] = entity_names[triple[2]]

        entries.append(entry)
        
    filename = os.path.join(dataset_path, "{}.json".format(dataset))
    try:
        logger.info("Saving wr18nn triples as {}".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            
        logger.info("Data saved to {}".format(filename))
            
    except Exception as e:
        global error_count
        error_count += 1 
        print("Data could not be saved as .json")
        print(e)


################################################################################################ 

def _load_wiki5m_entity_names(path: str) -> dict:
    wiki5m_entity_names = {}

    try:
        with open(path, "r", encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                split_line = line.strip().split("\t")
                entity_name = " ".join(split_line[1:])
                wiki5m_entity_names[split_line[0]] = " ".join(entity_name.split()[:10])
            logger.info("Wikidata5m_entity.txt loaded successfully ({})".format(args.task))

    except:
        global error_count
        error_count += 1
        logger.error("Wikidata5m_entity.txt could not be loaded successfully")
        logger.error(traceback.print_exc())
    
    return wiki5m_entity_names


def _load_wiki5m_entity_descriptions(path: str) -> dict:
    wiki5m_entity_descriptions = {}

    try:
        with open(path, "r", encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                split_line = line.strip().split("\t")
                description = " ".join(split_line[1:])
                wiki5m_entity_descriptions[split_line[0]] = " ".join(description.split()[:30])
        logger.info("Wikidata5m_text.txt loaded successfully ({})".format(args.task))

    except:
        global error_count
        error_count += 1
        logger.error("Wikidata5m_text.txt could not be loaded successfully")
        logger.error(traceback.print_exc())

    return wiki5m_entity_descriptions


def _load_wiki5m_relations(path: str) -> dict:
    wiki5m_relations = {}

    try:
        with open(path, "r", encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                split_line = line.strip().split("\t")
                relation = " ".join(split_line[1:])
                wiki5m_relations[split_line[0]] = " ".join(relation.split()[:10])
        logger.info("Wikidata5m_relation.txt loaded successfully ({})".format(args.task))

    except:
        global error_count
        error_count += 1
        logger.error("Wikidata5m_relation.txt could not be loaded successfully")
        logger.error(traceback.print_exc())

    return wiki5m_relations



def _load_wiki5_triples(dataset_path, dataset):
    triples = []
    path = os.path.join(dataset_path, "{}.txt".format(dataset))
    
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                triples.append(line.strip().split('\t'))
                
        logger.info("{} Wikidata5m triples loaded successfully".format(len(triples)))
        
    except Exception as e:
        global error_count
        error_count += 1
        logger.error("Wikidata5m data could not be loaded successfully")
        logger.error(traceback.print_exc())
             
    return triples


def wiki5m_to_json(triples: list, wiki5m_entity_names: dict, wiki5m_relations: dict, dataset_path: str, dataset: str) -> None:
    entries = []

    for triple in triples:
        entry = {}
        entry["head_id"] = triple[0]
        entry["head"] = wiki5m_entity_names.get(triple[0], None)
        entry["relation"] = wiki5m_relations.get(triple[1], None)
        entry["tail_id"] = triple[2] 
        entry["tail"] = wiki5m_entity_names.get(triple[2], None)

        entries.append(entry)

    invalid_triples = [entry for entry in entries if _has_none_value(entry)]
    logger.info('Found {} invalid triples in {}/{}'.format(len(invalid_triples), dataset_path, dataset))

    if dataset == "train":
        entries = [entry for entry in entries if not _has_none_value(entry)]
        
    filename = os.path.join(dataset_path, "{}.json".format(dataset))
    try:
        logger.info("Saving Wikidata5m triples as {}".format(filename)) 
        with open(filename, "w", encoding="utf-8") as out_file:
            json.dump(entries, out_file, indent = 4, ensure_ascii=False)
            
        logger.info("Data saved to {}".format(filename))
            
    except Exception as e:
        global error_count
        error_count += 1 
        print("Wikidata5m could not be saved as .json")
        print(e)


def _save_relations(relations: dict, data_dir) -> None:
    filepath = os.path.join(data_dir, "relations.json")
    try:
        with open(filepath, "w", encoding="utf-8") as out_file:
            json.dump(relations, out_file, ensure_ascii=False, indent=4)
            logger.info("Save {} realtions to \"{}\"".format(len(relations), filepath))

    except Exception as e:
        global error_count
        error_count += 1 
        logger.warning("Relations could not be saved")
        logger.warning(traceback.print_exc())


def _has_none_value(ex: dict) -> bool:
    return any(v is None for v in ex.values())


#################################################################################################


# FB15k_237 entity descriptions
fbk15_237_entity_descriptions = {}
error_count = 0
def main():
    
    # change directory, so the script will find the data
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath) 
    os.chdir(dname)
    
    logger.info("Start pre-processing") 
    
    datasets = ["train", "valid", "test"]
    
    dataset_path = os.path.join("..", "data", args.task)
    
    # list containing a dictionary of each triple in the data. The dictionary containes
    # the head_id, the head_name, the normalized relation, the tail_id and the tail name
    all_triples = []
    
    # load entity descriptions as dictionary with the the code as key and the descriptions as value
    if args.task.lower() == "fb15k237":

        descriptions_path = os.path.join(dataset_path, "FB15k_mid2description.txt")
        entity_descriptions = _load_fbk15_237_mid2descriptions(descriptions_path)

        mid2names_path = os.path.join(dataset_path, "FB15k_mid2name.txt")
        entity_names = _load_fbk15_237_mid2names(mid2names_path)
    
        for dataset in datasets:
            triples = _load_fbk15_237_triples(dataset_path, dataset)
            fbk15_237_to_json(triples, entity_names, dataset_path, dataset)
            all_triples += triples
    
        _save_entities_to_json(all_triples, entity_names, entity_descriptions, dataset_path)

    elif args.task.lower() == "wn18rr":

        wn18rr_names_and_descriptions_path = os.path.join(dataset_path, "wordnet-mlj12-definitions.txt")
        wn18rr_names_and_entity_descriptions = _load_wn18rr_descriptions(wn18rr_names_and_descriptions_path)
       
        wn18rr_entity_names, wn18rr_entity_descriptions = _separate_wr18nn_names_and_description(wn18rr_names_and_entity_descriptions)

        wn18rr_relations_path = os.path.join(dataset_path, "relations.dict")
        wn18rr_relations = _load_wn18rr_relations(wn18rr_relations_path)
        _save_relations(wn18rr_relations, dataset_path)

        for dataset in datasets:
            triples = _load_wn18rr_triples(dataset_path, dataset)
            wn18rr_to_json(triples, wn18rr_entity_names, wn18rr_relations, dataset_path, dataset)
            all_triples += triples

        _save_entities_to_json(all_triples, wn18rr_entity_names, wn18rr_entity_descriptions, dataset_path)
        

    elif args.task.lower() in ['wiki5m_trans', 'wiki5m_ind']:

        wiki5m_descriptions_path = os.path.join(dataset_path, "wikidata5m_text.txt")
        wiki5m_entity_descriptions = _load_wiki5m_entity_descriptions(wiki5m_descriptions_path)

        wiki5m_entity_path = os.path.join(dataset_path, "wikidata5m_entity.txt")
        wiki5m_entity_names = _load_wiki5m_entity_names(wiki5m_entity_path)

        wiki5m_relations_path = os.path.join(dataset_path, "wikidata5m_relation.txt")
        wiki5m_relations = _load_wiki5m_relations(wiki5m_relations_path)
        _save_relations(wiki5m_relations, dataset_path)

        for dataset in datasets:
            triples = _load_wiki5_triples(dataset_path, dataset)
            wiki5m_to_json(triples, wiki5m_entity_names, wiki5m_relations, dataset_path, dataset)
            all_triples += triples

        _save_entities_to_json(all_triples, wiki5m_entity_names, wiki5m_entity_descriptions, dataset_path)
    

    logger.info("Finished pre-processing with {} errors".format(error_count))


if __name__ == "__main__":
    main()