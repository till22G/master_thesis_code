import json
import time 
import numpy as np

def print_entities(entities: list):
    for item in entities:
        print("entity_id: {}".format(item["entity_id"]))
        print("entity: {}".format(item["entity"]))
        print(60 * "-")

# Open the JSON file for reading
with open('../SimKGC/SimKGC/data/FB15k237/entities.json', 'r', encoding='utf-8') as json_file:
    # Load the JSON data into a Python list
    data_list_1 = json.load(json_file)

# Now, 'data_list' contains the loaded JSON data as a list
print(len(data_list_1))

# Open the JSON file for reading
with open('data/FB15K237/preprocessed_entities.json', 'r', encoding='utf-8') as json_file:
    # Load the JSON data into a Python list
    data_list_2 = json.load(json_file)

# Now, 'data_list' contains the loaded JSON data as a list
print(len(data_list_2))

#data_list_2 = data_list_2[:1000]
#data_list_1 = data_list_1[:1000]

start = time.time()

mask = np.empty(len(data_list_2))
for i, item2 in enumerate(data_list_2):
    contained = False 
    for item1 in data_list_1:
        if item1["entity_id"] == item2["entity_id"]:
            contained = True
            break

    mask[i] = contained
                    
end = time.time()

mask_not_contained = np.logical_not(mask)
not_contained = np.array(data_list_2)[mask_not_contained]
print_entities(not_contained)
print("used time: {:.4f} seconds".format(end - start))
print(np.sum(mask_not_contained)) 