all_rust_input_path = "../data/translate/all_rust_input.txt"
all_rule_output_path = "../data/translate/all_rule_output.txt"


with open(all_rust_input_path, "r") as f_rust_input:
    all_rust_input_data = f_rust_input.readlines()

with open(all_rule_output_path, "r") as f_rule_output:
    all_rule_output_data = f_rule_output.readlines()

existing_output = []
existing_input = []
for i, output_data in enumerate(all_rule_output_data):
    if output_data not in existing_output:
        existing_output.append(output_data)
        existing_input.append(all_rust_input_data[i])
    # line = input_data + " " + all_rule_output_data[i]
    # all_lines.append(line)

print(len(existing_output))
# all_lines = list(set(all_lines))
# print(len(all_lines))