import random 

input_rust_path = "../new_data/all/input_rust.txt"
rule_path =  "../new_data/all/output_rule.txt"

with open(input_rust_path, "r") as f1:
    input_lines = f1.readlines()

with open(rule_path, "r") as f2:
    rule_lines = f2.readlines()


samples = list(zip(input_lines, rule_lines))

print(len(samples))
samples_no_duplicates = list(set(samples))
print(len(samples_no_duplicates))

random.shuffle(samples_no_duplicates)

# for input_line in input_lines:
#     for rule_line in rule_lines:
#         sample = zip(input_line, rule_line)
#         samples.append(sample)


train_count = int((len(samples_no_duplicates)*80)/100)
test_count = int((len(samples_no_duplicates)*20)/100)

train_start_index = 0
train_end_index = train_count

test_start_index = train_end_index 
test_end_index = len(input_lines)


train_samples = samples_no_duplicates[train_start_index:train_end_index]
test_samples = samples_no_duplicates[test_start_index:test_end_index]

train_rusts, train_rules = zip(*train_samples)
test_rusts, test_rules = zip(*test_samples)
# train_inputs = input_lines[train_start_index:train_end_index]
# test_inputs = input_lines[test_start_index:test_end_index]

# train_rules = rule_lines[train_start_index:train_end_index]
# test_rules = rule_lines[test_start_index:test_end_index]

print(len(train_rusts))

with open("../new_data/all/train_input_rust.txt", "w") as f_input_train:
    for i, train_input in enumerate(train_rusts):
        f_input_train.write(str(i) + "," + train_input)
        # f_input_train.write("\n")

print("DDD")
with open("../new_data/all/test_input_rust.txt", "w") as f_input_test:
    for i, test_input in enumerate(test_rusts):
        f_input_test.write(str(i) + "," + test_input)
        # f_input_test.write("\n")

with open("../new_data/all/train_rule.txt", "w") as f_rule_train:
    for i, train_rule in enumerate(train_rules):
        f_rule_train.write(str(i) + "," + train_rule)
        # f_rule_train.write("\n")

with open("../new_data/all/test_rule.txt", "w") as f_rule_test:
    for i, test_rule in enumerate(test_rules):
        f_rule_test.write(str(i) + "," + test_rule)
        # f_rule_test.write("\n")
