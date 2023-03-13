import json 
from transformers import RobertaTokenizer

input_path = "../processed_new_data/data_with_hole_in_context_train.json"

f = open(input_path)
data = json.load(f)


tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")

new_data = []
for sample in data:
    source = sample["before"] + " <unk> " + sample["after"] + " <unk> " + sample["context"]

    source_ids = tokenizer.encode(source)
    print(len(source_ids))
    if len(source_ids) < 2048:
        new_data.append(sample)

print("Length : ", len(new_data))
with open("../processed_new_data/data_with_hole_in_context_train_length_2048.json", "w", encoding='utf-8') as f_input_train:
    json.dump(new_data, f_input_train, ensure_ascii=False, indent=4)