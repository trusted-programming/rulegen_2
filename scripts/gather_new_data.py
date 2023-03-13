import os
import re
path = "../new_data_raw/triple_folder1"
before_output_path =  "../new_data/all/before.txt"
after_output_path =  "../new_data/all/after.txt"
input_rust_path =  "../new_data/all/input_rust.txt"
output_rule_path =  "../new_data/all/output_rule.txt"
sample_dirs = os.listdir(path)

for dir in sample_dirs:
    print("----------------")
    dir_path = os.path.join(path,dir)
    before_path = os.path.join(dir_path,"before")
    after_path = os.path.join(dir_path,"after")
    rule_path = os.path.join(dir_path,"rule")

    print(before_path)
    with open(before_path, "r") as f_before:
        before_data = str(f_before.read())
    
    with open(after_path, "r") as f_after:
        after_data = str(f_after.read())

    with open(rule_path, "r") as f_rule:
        rule_data = str(f_rule.read())
    
    before_data = " ".join(before_data.split())
    after_data = " ".join(after_data.split())
    rule_data = " ".join(rule_data.split())

    input_rust = before_data + " <unk> " + after_data

    # if ";" in before_data or "}" in before_data or "{" in before_data:
    # if re.search("[;{}=]", before_data) and re.search("[;{}=]", after_data) and before_data and after_data and rule_data:
    if before_data and after_data and rule_data:
        with open(before_output_path, "a") as f1:
            f1.write(before_data)
            f1.write("\n")

        with open(after_output_path, "a") as f2:
            f2.write(after_data)
            f2.write("\n")
        
        with open(input_rust_path, "a") as f3:
            f3.write(input_rust)
            f3.write("\n")
        
        with open(output_rule_path, "a") as f4:
            f4.write(rule_data)
            f4.write("\n")
