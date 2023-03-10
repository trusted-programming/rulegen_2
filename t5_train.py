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
            "before": sample["before"],
            "after": sample["after"]
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

    predict_nls = []
    golds = []
    

    results = []
    for step, batch in enumerate(bar):
        batch = tuple(t.to(device) for t in batch)
        idx, source_ids, target_ids = batch
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
            print(predict_nl)
            sample_index = str(idx[i])

            predict_nl = (sample_index + '\t' + predict_nl_origin.strip() + '\n')
            gold = (sample_index + '\t' + eval_samples[sample_index]["after"].strip() + '\n')
            print(gold)
            predict_nls.append(predict_nl)
            golds.append(gold)

            result_obj = {
                "before": eval_samples[sample_index]["before"],
                "after": eval_samples[sample_index]["after"],
                "predict": predict_nl_origin
            }

            results.append(result_obj)
            # with open("test_result.json", "a") as f_result:
                
    with open("test_results.json", 'w') as f_result:
        json.dump(results, f_result)

    (goldMap, predictionMap) = computeMaps(predict_nls, golds)
    bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
    print("BlEU : ", bleu)
    return bleu

def train(config, device, model, encoder_config, train_dataloader, val_dataloader):
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

    multi_gpu_training = config.getboolean("multi_gpu_training")
    
    num_train_optimization_steps = num_train_epochs * len(train_dataloader)
    save_steps = max(len(train_dataloader), 1)
    # save_steps = 50

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    if warmup_steps < 1:
        warmup_steps = num_train_optimization_steps * warmup_steps
    else:
        warmup_steps = warmup_steps

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    best_bleu = 0.0
    # evaluate(config, device, model, encoder_config, val_dataloader)
    # evaluate(config, device, model, encoder_config, val_dataloader)
    global_step = 0
    for epoch in range(start_epoch, num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        # nb_tr_steps = 0
        # tr_loss = 0.0

        if multi_gpu_training:
            logger.info("Set epoch for sampler...")
            train_dataloader.sampler.set_epoch(epoch)
            
        model.train()


        for step, batch in enumerate(bar):
            batch = tuple(t.to(device) for t in batch)
            _, source_ids, target_ids = batch
            eos_mask = source_ids.eq(encoder_config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                print("Skipping this batch...")
                continue

            loss = model(source_ids=source_ids,
                        target_ids=target_ids)
            
            if multi_gpu_training:
                loss = loss.mean() # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            # print(loss.item())
            # tr_loss += loss.item()
            # nb_tr_steps += 1

            loss.backward()

            # if nb_tr_steps % args.gradient_accumulation_steps == 0:
                # Update parameters
                # optimizer.step()
                # optimizer.zero_grad()
                # scheduler.step()
                # global_step += 1
                # train_loss = round(tr_loss * gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                # bar.set_description("[{}] Train loss {} best B {}".format(epoch, round(train_loss.item(), 3), round(best_bleu,3)))

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            bar.set_description("[{}] Train loss {} best B {}".format(epoch, round(loss.item(), 3), round(best_bleu,3)))

            dist.barrier()
            if (step + 1) % save_steps == 0:
        # if epoch > 10:
                if ddp_utils.is_main_process():
                    if not os.path.exists(checkpoint_path):
                        os.mkdir(checkpoint_path)
                    
                    output_model_file = os.path.join(checkpoint_path, f"{checkpoint_name}_epoch_{epoch}.bin")
                    
                    # bleu = evaluate(config, device, model, encoder_config, val_dataloader)

                    # if bleu > best_bleu:
                        # best_bleu = bleu
                    torch.save(model.state_dict(), output_model_file)
                    logger.info("Save the model into %s", output_model_file)
                

    cleanup()


def init_training_dataloader(config):
    train_data_processor = RuleDataProcessor(tokenizer_path=config["tokenizer_name"], model_config_path=config["config_path"],
                                            data_path=config["train_path"], 
                                            output_cache_path=config["train_cache_path"], 
                                            max_source_length=config.getint("max_source_length"), 
                                            max_target_length=config.getint("max_target_length"))
    
    # train_data_processor = RuleDataProcessor(tokenizer_path=config["tokenizer_name"], model_config_path=config["config_path"],
    #                                         rust_input_path=config["val_rust_input_path"], 
    #                                         rule_output_path=config["val_rule_output_path"],
    #                                         output_cache_path=config["val_cache_path"], 
    #                                         max_source_length=config.getint("max_source_length"), 
    #                                         max_target_length=config.getint("max_target_length"))

    train_dataset = train_data_processor.dataset

    if config.getboolean("multi_gpu_training"):     
        logger.info("*** Using DistributedSampler ***")
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        logger.info("*** Using RandomSampler ***")
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.getint("batch_size"), num_workers=config.getint("num_workers"), pin_memory=False)
    return train_dataloader


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



def setup():
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29514'

    dist_url = "env://"
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # initialize the process group
    dist.init_process_group(backend="nccl", init_method=dist_url, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    dist.barrier()

    ddp_utils.setup_for_distributed(rank == 0)

def cleanup():
    dist.destroy_process_group()

def run_spawn(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    #----------------------------
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
    # n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    # world_size = n_gpus
    # run_spawn(main, world_size)
    #----------------------------

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    # local_rank = 0
    # local_rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup()
    config_ini_path = "config/t5_config.ini"
    configs = configparser.ConfigParser()
    configs.read(config_ini_path)

    config = configs["neural_network"]

    config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
    encoder_config = config_class.from_pretrained(config["config_path"])
    pretrained_model = model_class.from_pretrained(config["model_name"])
    tokenizer = tokenizer_class.from_pretrained(config["tokenizer_name"])


    checkpoint_path = config["checkpoint_path"]
    setting = config.getint("setting")
    checkpoint_name = config["checkpoint_name"]
    start_epoch = config.getint("start_epoch")
    
    train_dataloader = init_training_dataloader(config)
    val_dataloader = init_val_dataloader(config)
    # val_dataloader = train_dataloader
    model = T5RuleGenerationModel(t5_model=pretrained_model, config=encoder_config, tokenizer=tokenizer, 
                                batch_size=config.getint("batch_size"), 
                                max_source_length=config.getint("max_source_length"), 
                                max_target_length=config.getint("max_target_length"), 
                                beam_size=config.getint("beam_size"))

    local_rank = int(os.environ["LOCAL_RANK"])

    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)


    # existing_model_checkpoint = os.path.join(checkpoint_path, f"{checkpoint_name}_epoch_{start_epoch}.bin")
    # if os.path.exists(existing_model_checkpoint):
    #     # start_epoch += 1
    #     configs.set("neural_network", "start_epoch", str(start_epoch+1))
    #     logger.info("*** Resume training from checkpoints ***")
    #     logger.info("Model checkpoint : %s ", existing_model_checkpoint)
    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    #     model.load_state_dict(torch.load(existing_model_checkpoint,  map_location=map_location))
        # model.load_state_dict(torch.load(existing_model_checkpoint))

    train(configs["neural_network"], local_rank, model, encoder_config, train_dataloader, val_dataloader)

    # evaluate(config, device, model, encoder_config, val_dataloader)

    # main()

