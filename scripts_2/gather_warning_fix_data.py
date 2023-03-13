import os
path = "../warning_fix_data/clippy-warning-fix/"

files = os.listdir(path)

patterns = []
for file in files:
    file_name = file.split(".")[0]
    if file_name != "clippy":
        patterns.append(file_name)

patterns = list(set(patterns))
pairs = []
for pattern in patterns:
    warning_name = pattern + ".cs-java.txt.cs"
    fix_name = pattern + ".cs-java.txt.java"
    
    warning_path = os.path.join(path, warning_name)
    fix_path = os.path.join(path, fix_name)

    with open(warning_path, "r") as f_warning:
        warning_data = f_warning.readlines()

    with open(fix_path, "r") as f_fix:
        fix_data = f_fix.readlines()

    for i, warning_line in enumerate(warning_data):
        if warning_line:
            obj = {}
            obj["index"] = i
            obj["before"] = warning_line.replace("\n","")
            obj["after"] = fix_data[i].replace("\n","")
            obj["pattern"] = pattern
            pairs.append(obj)


output_path = "../warning_fix_data/all_pairs.json"
import json
with open(output_path, 'w') as f:
    json.dump(pairs, f)

# print(pairs)

   