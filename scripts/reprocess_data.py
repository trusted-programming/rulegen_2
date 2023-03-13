train_rust_input_path = "../data/translate/train_rust_input.txt"
train_rule_output_path = "../data/translate/train_rule_output.txt"

test_rust_input_path = "../data/translate/test_rust_input.txt"
test_rule_output_path = "../data/translate/test_rule_output.txt"

val_rust_input_path = "../data/translate/val_rust_input.txt"
val_rule_output_path = "../data/translate/val_rule_output.txt"


with open(train_rust_input_path, "r") as f_train_rust_input:
    train_rust_input_data = f_train_rust_input.readlines()

with open(train_rule_output_path, "r") as f_train_rule_output:
    train_rule_output_data = f_train_rule_output.readlines()

with open(test_rust_input_path, "r") as f_test_rust_input:
    test_rust_input_data = f_test_rust_input.readlines()

with open(test_rule_output_path, "r") as f_test_rule_output:
    test_rule_output_data = f_test_rule_output.readlines()

with open(val_rust_input_path, "r") as f_val_rust_input:
    val_rust_input_data = f_val_rust_input.readlines()

with open(val_rule_output_path, "r") as f_val_rule_output:
    val_rule_output_data = f_val_rule_output.readlines()

all_rust_input_data = train_rust_input_data + test_rust_input_data + val_rust_input_data
all_rule_output_data = train_rule_output_data + test_rule_output_data + val_rule_output_data

with open("../data/translate/all_rust_input.txt", "w") as f:
    for input_rust in all_rust_input_data:
        f.write(input_rust)
        # f.write("\n")

with open("../data/translate/all_rule_output.txt", "w") as f1:
    for output_rule in all_rule_output_data:
        f1.write(output_rule)
        # f1.write("\n")
