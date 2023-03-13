import json 
import os
import random
input_path = "../processed_new_data/data_with_hole_in_context_test.json"

f = open(input_path)
data = json.load(f)

random.shuffle(data)

data = data[:100]

output_root_path = "../demo_samples"
for i, sample in enumerate(data):

    if not os.path.exists(os.path.join(output_root_path, str(i))):
        os.mkdir(os.path.join(output_root_path, str(i)))

    before_path = os.path.join(output_root_path, str(i), "before.txt")
    after_path = os.path.join(output_root_path, str(i), "after.txt")
    context_path = os.path.join(output_root_path, str(i), "context.txt")
    hole_rule_path = os.path.join(output_root_path, str(i), "hole_rule.txt")

    with open(before_path, "w") as f_before:
        f_before.write(sample["before"])
    
    with open(after_path, "w") as f_after:
        f_after.write(sample["after"])
    
    with open(context_path, "w") as f_context:
        f_context.write(sample["context"])
    
    with open(hole_rule_path, "w") as f_hole_rule:
        f_hole_rule.write(sample["hole_rule"])

# with open("../processed_new_data/data_with_hole_in_context_train_small_100.json", "w", encoding='utf-8') as f_input_train:
#     json.dump(data_small, f_input_train, ensure_ascii=False, indent=4)