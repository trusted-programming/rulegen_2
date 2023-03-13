import random 
import json


input_path = "../processed_new_data/data_with_hole_in_context.json"

f = open(input_path)
data = json.load(f)

random.shuffle(data)

train_count = int((len(data)*90)/100)
test_count = int((len(data)*10)/100)

train_start_index = 0
train_end_index = train_count

test_start_index = train_end_index 
test_end_index = len(data)

train_samples = data[train_start_index:train_end_index]
test_samples = data[test_start_index:test_end_index]

with open("../processed_new_data/data_with_hole_in_context_train.json", "w", encoding='utf-8') as f_input_train:
    json.dump(train_samples, f_input_train, ensure_ascii=False, indent=4)

with open("../processed_new_data/data_with_hole_in_context_test.json", "w", encoding='utf-8') as f_input_test:
    json.dump(test_samples, f_input_test, ensure_ascii=False, indent=4)