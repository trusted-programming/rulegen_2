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
# from sacrebleu.metrics import BLEU
from collections import OrderedDict
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# from eval_utils import generate_fixes
# from eval_utils import evaluate
from model.t5_rule_generation_model import T5RuleGenerationModel
import configparser
import argparse



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)





if __name__ == "__main__":

    # Instantiate the argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--before_path", type=str,
                        help="Original code snippet")
    
    parser.add_argument("--after_path", type=str,
                        help="Transformed code snippet")
    
    parser.add_argument("--context_path", type=str,
                        help="Context of existing rules")

    # parser.add_argument("--missing_rule_path", type=str,
    #                     help='Path to save the missing rule')

    # before is A
    # after is B
    # context is C
    # Apply C(A)!= B ==> need prediction
    # After prediction, get C', 

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
    local_rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_ini_path = "config/t5_config.ini"
    configs = configparser.ConfigParser()
    configs.read(config_ini_path)

    config = configs["neural_network"]

    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["checkpoint_name"]

    config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
    encoder_config = config_class.from_pretrained(config["config_path"])
    pretrained_model = model_class.from_pretrained(config["model_name"])
    tokenizer = tokenizer_class.from_pretrained(config["tokenizer_name"])


    model = T5RuleGenerationModel(t5_model=pretrained_model, config=encoder_config, tokenizer=tokenizer, 
                                batch_size=config.getint("batch_size"), 
                                max_source_length=config.getint("max_source_length"), 
                                max_target_length=config.getint("max_target_length"), 
                                beam_size=config.getint("beam_size"))
    
    model.to(local_rank)

    existing_model_checkpoint = config["pretrained_model_path"]
    
    if os.path.exists(existing_model_checkpoint):
        logger.info("*** Resume training from checkpoints ***")
        logger.info("Model checkpoint : %s ", existing_model_checkpoint)
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        # model.load_state_dict(torch.load(existing_model_checkpoint,  map_location=map_location))
        # model.load_state_dict(torch.load(existing_model_checkpoint))
        state_dict = torch.load(existing_model_checkpoint, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


        with open(args.before_path, "r") as f1:
            before = f1.read()
        
        with open(args.after_path, "r") as f2:
            after = f2.read()

        with open(args.context_path, "r") as f3:
            context = f3.read()

        # source = " ".join(source.split())
        # target = " ".join(target.split())
        
        source_input = f"{before} <unk> {after} <unk> {context}"
        # print(rust_input)
        source_input_ids = tokenizer.encode(source_input, max_length=config.getint("max_source_length"), padding='max_length', truncation=True)
        # print(rust_input_ids)
        source_input_ids = torch.tensor([source_input_ids])
        source_input_ids = source_input_ids.to(local_rank)
        
        predict_ids = model(source_ids=source_input_ids, generate_target=True)
        # print(predict_ids)
        predict_nl = tokenizer.decode(predict_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


        print(predict_nl)
    else:
        logger.info("*** Path does not exists!!!! ***")

  
    
    # main()

