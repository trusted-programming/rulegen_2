import re
import json
import os
# path = "../warning_fix_data_2/clippy-warning-fixes-function/test.cs"
sample_path =  "../warning_fix_data_2/clippy-warning-fixes-function/##[Warning(bare_trait_objects).cs-java.txt.cs"
# text = "##[Warning(bare_trait_objects)\n@@ -74,5 +102,9 @@ impl<F> Future for Shared<F>\n\n##[Warning(bare_trait_objects)\n@@ -84,8 +88,10 @@ impl<I> Future for JoinAll<I>\n\n##[Warning(bare_trait_objects)\n@@ -33,0 +35 @@ pub struct BiLock<T> {\npub type LocalMap = RefCell<HashMap<TypeId,\n                                    Box<Opaque>,\n                                    BuildHasherDefault<IdHasher>>>;\n##[Warning(bare_trait_objects)\n@@ -25,17 +23,0 @@ pub struct Buffered<S>"
path = "../warning_fix_data_2/clippy-warning-fixes-function/"
train_output_path = "../warning_fix_data_2/all_pairs.json"
files = os.listdir(path)
pattern = re.compile("@@.*@@")

def extract_samples(content):
    # print(content)
    lines = content.split("\n")
    results = []
    temp = ""
    last_warning = False
    for line in content.split("\n"):
        if line.startswith("##[Warning"):
            if temp or last_warning:
                results.append(temp.strip())
            temp = ""
            last_warning = True
        else:
            temp += line + "\n"
            last_warning = False

    if temp:
        temp = temp.strip()
        temp = re.sub(r"^\s*///.*$", "", temp, flags=re.MULTILINE)
        temp = "\n".join([line[:len(line) - len(line.lstrip())] + line.lstrip() for line in temp.split("\n") if line.strip() != ""])
        results.append(temp.strip())

    # if temp:
    #     print(temp)
    #     temp = " ".join(temp)
    #     f = re.sub(r"^\s*///.*$", "", temp, flags=re.MULTILINE)
    #     f = "\n".join([line[:len(line) - len(line.lstrip())] + line.lstrip() for line in temp.split("\n") if line.strip() != ""])
    #     final_result.append(f)

    return results

patterns = []
for file in files:
    file_name = file.split(".")[0]
    # if  "clippy" not in file_name:
    patterns.append(file_name)

patterns = list(set(patterns))
pairs = []
# pairs = []
for pattern in patterns:
    print("---------------")
    print(pattern)
    warning_name = pattern + ".cs-java.txt.cs"
    fix_name = pattern + ".cs-java.txt.java"
    
    warning_path = os.path.join(path, warning_name)
    fix_path = os.path.join(path, fix_name)

    with open(warning_path, "r") as f_warning:
        warning_data = f_warning.read()
    
    with open(fix_path, "r") as f_fix:
        fix_data = f_fix.read()

    warning_samples = extract_samples(warning_data)
    fix_samples = extract_samples(fix_data)
    print(len(warning_samples))
    print(len(fix_samples))
    assert len(warning_samples) == len(fix_samples)
    for i, warning_sample in enumerate(warning_samples):
        if warning_sample and fix_samples[i]:
            obj = {}
            obj["index"] = i
            obj["before"] = warning_sample
            obj["after"] = fix_samples[i]
            obj["pattern"] = pattern
            pairs.append(obj)


print(len(pairs))
with open(train_output_path, 'w') as f_train:
    json.dump(pairs, f_train)
# with open(sample_path, "r") as f:
#     data = f.read()

# # print(extract_samples(data))
# for f in extract_samples(data):
#     print("----------")
#     print(f)
