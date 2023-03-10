from data_processor.rule_processor import RuleDataProcessor
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, DistributedSampler, TensorDataset
import multiprocessing
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import torch
import torch.nn as nn
import logging
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
# from utils import smooth_bleu
from sacrebleu.metrics import BLEU
from collections import OrderedDict
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# from eval_utils import generate_fixes
# from eval_utils import evaluate
from model.t5_rule_generation_model import T5RuleGenerationModel
import configparser
import ddp_utils
from evaluator import computeMaps
from evaluator import bleuFromMaps
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_golds(path):
    f = open(path)
    data = json.load(f)
    gold_dict = {}
    
    # lines = []
    # for sample in data:
    #     lines.append(sample["after"])
   
    # with open(path, "r") as f:
    #     lines = f.readlines()
    for sample in data:
        # line_split = line.split(",")
        # gold_dict[sample["index"]] = sample["after"]
        gold_dict[sample["index"]] = {
            "pattern": sample["pattern"],
            "before": sample["before"],
            "after": sample["after"],
        }
    return gold_dict



def evaluate(config, device, model, encoder_config, test_dataloader):

    setting = config.getint("setting")
    batch_size = config.getint("batch_size")
    learning_rate = config.getfloat("learning_rate")
    adam_epsilon = config.getfloat("adam_epsilon")
    warmup_steps = config.getint("warmup_steps")
    max_source_length = config.getint("max_source_length")
    max_target_length = config.getint("max_target_length")
    beam_size = config.getint("beam_size")
    num_labels = config.getint("num_labels")
    num_workers = config.getint("num_workers")
    weight_decay = config.getfloat("weight_decay")
    max_grad_norm = config.getfloat("max_grad_norm")
    gradient_accumulation_steps = config.getint("gradient_accumulation_steps")
    start_epoch = config.getint("start_epoch")
    num_train_epochs = config.getint("num_train_epochs")

    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["checkpoint_name"]
    
    val_path = config["val_path"]

    bar = tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluating")
    
    eval_samples = load_golds(val_path)
    print(eval_samples.keys())

    predict_nls = []
    golds = []
    
    results = []
    for step, batch in enumerate(bar):
        batch = tuple(t.to(device) for t in batch)
        idx, source_ids, target_ids = batch
        # print(idx)
        idx = list(idx.cpu().numpy())
        predict_ids = model(source_ids=source_ids,
                                target_ids=target_ids,
                                generate_target=True)
        # print(predict_ids)

        top_preds = list(predict_ids.cpu().numpy())

        # print(top_preds)
        
        for i, top_pred in enumerate(top_preds):
            print("--------------------")
            predict_nl_origin = tokenizer.decode(top_pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(predict_nl_origin)
            sample_index = idx[i]

            predict_nl = (str(sample_index) + '\t' + predict_nl_origin.strip() + '\n')
            gold = (str(sample_index) + '\t' + eval_samples[sample_index]["after"].strip() + '\n')
            print(gold)
            predict_nls.append(predict_nl)
            golds.append(gold)

            result_obj = {
                "warning_type": eval_samples[sample_index]["pattern"],
                "before": eval_samples[sample_index]["before"],
                "after": eval_samples[sample_index]["after"],
                "predict": predict_nl_origin
            }

            results.append(result_obj)
    
    with open("test_results.json", 'w') as f_result:
        json.dump(results, f_result)

    (goldMap, predictionMap) = computeMaps(predict_nls, golds)
    bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
    print("BlEU : ", bleu)
    return bleu

def init_val_dataloader(config):
    val_data_processor = RuleDataProcessor(tokenizer_path=config["tokenizer_name"], model_config_path=config["config_path"],
                                            data_path=config["val_path"], 
                                            output_cache_path=config["val_cache_path"], 
                                            max_source_length=config.getint("max_source_length"), 
                                            max_target_length=config.getint("max_target_length"))

    val_dataset = val_data_processor.dataset
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=config.getint("batch_size"), num_workers=config.getint("num_workers"), pin_memory=False)
    return val_dataloader



if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

    config_ini_path = "config/t5_config.ini"
    configs = configparser.ConfigParser()
    configs.read(config_ini_path)

    config = configs["neural_network"]

    config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
    encoder_config = config_class.from_pretrained(config["config_path"])
    pretrained_model = model_class.from_pretrained(config["model_name"])
    tokenizer = tokenizer_class.from_pretrained(config["tokenizer_name"])


    checkpoint_path = config["pretrained_model_path"]
    # setting = config.getint("setting")
    # checkpoint_name = config["checkpoint_name"]
    # start_epoch = config.getint("start_epoch")
    
    val_dataloader = init_val_dataloader(config)
    # val_dataloader = train_dataloader
    model = T5RuleGenerationModel(t5_model=pretrained_model, config=encoder_config, tokenizer=tokenizer, 
                                batch_size=config.getint("batch_size"), 
                                max_source_length=config.getint("max_source_length"), 
                                max_target_length=config.getint("max_target_length"), 
                                beam_size=config.getint("beam_size"))

    local_rank = 0
    model.to(local_rank)

    # existing_model_checkpoint = os.path.join(checkpoint_path, f"{checkpoint_name}_epoch_{start_epoch}.bin")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    evaluate(configs["neural_network"], local_rank, model, encoder_config, val_dataloader)

    # main()

