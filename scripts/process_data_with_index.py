

path = "../data/translate/val_rule_output.txt"
output_path = "../data/translate_2/val_rule_output.txt"
with open(path, "r") as f:
    data = f.readlines()

# lines = []
# for i, item in data:
#     line = str(i) + "," + item
#     lines.append(line)

with open(output_path, "w") as f1:
    for i, item in enumerate(data):
        line = str(i) + "," + item
        f1.write(line)


