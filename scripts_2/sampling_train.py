import json 

input_path = "../processed_new_data/data_with_hole_in_context_train.json"

f = open(input_path)
data = json.load(f)


data_small = data[:100]

with open("../processed_new_data/data_with_hole_in_context_train_small_100.json", "w", encoding='utf-8') as f_input_train:
    json.dump(data_small, f_input_train, ensure_ascii=False, indent=4)