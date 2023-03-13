before_path =  "../new_data/before.txt"
after_path =  "../new_data/after.txt"
# rule_path =  "../new_data/rule.txt"

with open(before_path, "r") as f1:
    before_lines = f1.readlines()

with open(after_path, "r") as f2:
    after_lines = f2.readlines()

# with open(rule_path, "r") as f3:
#     rule_lines = f3.readlines()



with open("../new_data/input_rust.txt", "w") as f:
    for i, before_line in enumerate(before_lines):
        after_line = after_lines[i].replace("\n","")
        before_line = before_line.replace("\n","")
        # rule_line = rule_lines
        line = before_line + " <unk> " + after_line
        f.write(line)
        f.write("\n")

