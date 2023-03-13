import random 
import json
  

all_pairs_path =  "../warning_fix_data_2/all_pairs.json"

f = open(all_pairs_path)
  
data = json.load(f)

random.shuffle(data)

train_count = int((len(data)*80)/100)
test_count = int((len(data)*20)/100)

train_start_index = 0
train_end_index = train_count

test_start_index = train_end_index 
test_end_index = len(data)

train_samples = data[train_start_index:train_end_index]
test_samples = data[test_start_index:test_end_index]

# print(test_samples)

train_output_path = "../warning_fix_data_2/train_pairs.json"
test_output_path = "../warning_fix_data_2/test_pairs.json"

with open(train_output_path, 'w') as f_train:
    json.dump(train_samples, f_train)

with open(test_output_path, 'w') as f_test:
    json.dump(test_samples, f_test)