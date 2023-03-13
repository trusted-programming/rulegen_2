import os
import re
import json
path = "../new_data_raw/triple_folder1"
# before_output_path =  "../new_data/all/before.txt"
# after_output_path =  "../new_data/all/after.txt"
# input_rust_path =  "../new_data/all/input_rust.txt"
# output_rule_path =  "../new_data/all/output_rule.txt"
output = "../new_data_2/all_data.json"
sample_dirs = os.listdir(path)


data_json = []

for i, dir in enumerate(sample_dirs):
    print("----------------")
    dir_path = os.path.join(path,dir)
    before_path = os.path.join(dir_path,"before")
    after_path = os.path.join(dir_path,"after")
    rule_path = os.path.join(dir_path,"rule")
    context_path = os.path.join(dir_path,"context")

    print(before_path)
    with open(before_path, "r") as f_before:
        befores_data = str(f_before.read())
    
    with open(after_path, "r") as f_after:
        after_data = str(f_after.read())

    with open(rule_path, "r") as f_rule:
        rule_data = str(f_rule.read())
    
    with open(context_path, "r") as f_context:
        context_data = str(f_context.read())
    
    # obj = {}
    # obj["index"] = i
    # obj["before"] = before_data
    # obj["after"] = after_data
    # obj["rule"] = rule_data
    # obj["context"] = context_data

    # before_data = " ".join(before_data.split())
    # after_data = " ".join(after_data.split())
    # rule_data = " ".join(rule_data.split())

    # input_rust = before_data + " <unk> " + after_data


    # if ";" in before_data or "}" in before_data or "{" in before_data:
    # if re.search("[;{}=]", before_data) and re.search("[;{}=]", after_data) and before_data and after_data and rule_data:
    if before_data and after_data and rule_data:
        obj = {}
        obj["index"] = i
        obj["before"] = before_data
        obj["after"] = after_data
        obj["rule"] = rule_data
        obj["context"] = context_data

        data_json.append(obj)

with open(output, 'w', encoding='utf-8') as f:
    json.dump(data_json, f, ensure_ascii=False, indent=4)
        # with open(before_output_path, "a") as f1:
        #     f1.write(before_data)
        #     f1.write("\n")

        # with open(before_output_path, "a") as f1:
        #     f1.write(before_data)
        #     f1.write("\n")

        # with open(after_output_path, "a") as f2:
        #     f2.write(after_data)
        #     f2.write("\n")
        
        # with open(input_rust_path, "a") as f3:
        #     f3.write(input_rust)
        #     f3.write("\n")
        
        # with open(output_rule_path, "a") as f4:
        #     f4.write(rule_data)
        #     f4.write("\n")
