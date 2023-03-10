from tqdm import *
import random
import logging
import json
import sys
import torch
import multiprocessing
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.data import TensorDataset
import concurrent.futures
import os
import numpy as np
import json
logger = logging.getLogger(__name__)

class Features(object):
    def __init__(self, sample_index, source_ids, target_ids):
        self.sample_index = sample_index
        self.source_ids = source_ids
        self.target_ids = target_ids

class RuleDataProcessor():
    
    def __init__(self, tokenizer_path, model_config_path, data_path: str, 
                output_cache_path: str, 
                max_source_length: str, max_target_length: str):
        self.data_path = data_path
        self.output_cache_path = output_cache_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        self.model_config = T5Config.from_pretrained(model_config_path)
        logger.info("Preparing dataloader.....")
        
        if os.path.exists(self.output_cache_path):
            print("Loading negative data from cache : ", self.output_cache_path)
            self.dataset = torch.load(self.output_cache_path)
        else:
            self.dataset = self.load_data()
            
    def load_data(self):
        f = open(self.data_path)
        print(self.data_path)
        data = json.load(f)

        # with open(self.data_path, "r") as f1:
        #     rust_inputs = f1.readlines()
            
        # with open(self.rule_output_path, "r") as f2:
        #     rule_outputs = f2.readlines()
            
        data_tuples = []
        for i, sample in enumerate(data):

            source = sample["before"] + "<unk>" + sample["context"]
            target = sample["after"]

            data_tuple = (i, source, sample["after"])
            data_tuples.append(data_tuple)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-5) as executor:
            features = list(executor.map(lambda f: self.convert_examples_to_features(*f), tuple(data_tuples)))	
        
        features = [feature for feature in features if feature != None]

        all_sample_ids_tensor = torch.tensor([f.sample_index for f in features], dtype=torch.long)
        all_source_token_ids_tensor = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_target_token_ids_tensor = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        all_data = TensorDataset(all_sample_ids_tensor, all_source_token_ids_tensor, all_target_token_ids_tensor)
        torch.save(all_data, self.output_cache_path)
        return all_data
        
    def convert_examples_to_features(self, index, source, target):

        # source_split = source.split(",")
        # target_split = target.split(",")

        # assert source_split[0] == target_split[0]

        # index = int(source_split[0])
        
        source_ids = self.tokenizer.encode(source, max_length=1000, padding='max_length', truncation=True)
        target_ids = self.tokenizer.encode(target, max_length=1000, padding='max_length', truncation=True)

        count_source_ids = [source_id for source_id in source_ids if source_id != 0]
        count_target_ids = [target_id for target_id in target_ids if target_id != 0]

        if len(count_source_ids) <= self.max_source_length and len(count_target_ids) <= self.max_source_length:
            # print(" ".join(rust_input_split[1:]))
            # print(" ".join(rule_output_split[1:]))
            source_ids = self.tokenizer.encode(source, max_length=self.max_source_length, padding='max_length', truncation=True)
            target_ids = self.tokenizer.encode(target, max_length=self.max_target_length, padding='max_length', truncation=True)

            # print(rule_output_ids)
            source_ids_2D = torch.tensor([source_ids])
            source_token_mask = source_ids_2D.eq(self.model_config.eos_token_id)
            
            target_ids_2D = torch.tensor([target_ids])
            target_token_mask = target_ids_2D.eq(self.model_config.eos_token_id)
            if len(torch.unique(source_token_mask.sum(1))) > 1 or len(torch.unique(target_token_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            
            return Features(sample_index=index, target_ids=target_ids, 
                                    source_ids=source_ids)
        
        else:
            return None
                                
    def _pad(self, arrays):
        max_batch = max([len(x) for x in arrays])
        arrays = [n + [0] * (max_batch - len(n)) for n in arrays]
        arrays = np.asarray(arrays)
        return arrays