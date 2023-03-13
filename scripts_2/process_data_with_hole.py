import os
import json 

path = "../new_data_06-03/triplets0221/dataset_with_holes/triplets_with_hole_in_context" #55562 instances

triplet_folders = os.listdir(path)


data_json = []

for triplet_folder in triplet_folders:
    triplet_folder_path = os.path.join(path, triplet_folder)
    
    triplet_data = os.listdir(triplet_folder_path)
    
    triplet = {}
    for triplet_file in triplet_data:
        triplet_file_path = os.path.join(triplet_folder_path, triplet_file)
        if triplet_file in ["before", "after", "context", "hole_rule"]:
            # triplets.append(triplet_file_path)
            with open(triplet_file_path, "r" , errors="ignore") as f:
                data = str(f.read())

            if triplet_file == "before":
                triplet["before"] = data
            elif triplet_file == "after":
                triplet["after"] = data
            elif triplet_file == "context":
                triplet["context"] = data
            elif triplet_file == "hole_rule":
                triplet["hole_rule"] = data
    
    # print(triplet)

    expected_keys = ["before", "after", "context", "hole_rule"]
    assert all(key in triplet for key in expected_keys), "Not all keys exist in triplet."

    data_json.append(triplet)

print("Num instances :", len(data_json)) ## 55562
with open("../processed_new_data/data_with_hole_in_context.json", 'w', encoding='utf-8') as f:
    json.dump(data_json, f, ensure_ascii=False, indent=4)