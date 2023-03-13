import json
import os
import lizard
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from transformers import RobertaTokenizer
import statistics


max_length = 200000

data_path = "../data/translate/val_rust_input.txt"

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

# Opening JSON file
with open(data_path, "r") as f:
    lines = f.readlines()
  
# returns JSON object as 

all_sample_lens = []
for item in lines:
    
    token_ids = tokenizer.encode(item, max_length=max_length, padding='max_length', truncation=True)
    
    token_ids = [token_id for token_id in token_ids if token_id != 0]
    all_sample_lens.append(len(token_ids))


print(len(all_sample_lens))
print("Median : ", statistics.median(all_sample_lens))
print("Max : ", max(all_sample_lens))
print("Min : ", min(all_sample_lens))
print("Mean : ", statistics.mean(all_sample_lens))

            