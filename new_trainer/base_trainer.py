
import torch
import torch.nn.functional as F
import torch.nn as nn
from accelerate import Accelerator,DistributedType
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,logging,set_seed
from omegaconf import OmegaConf
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig, PrefixTuningConfig, PromptTuningInit, PromptTuningConfig
from codetf.common.utils import get_abs_path
import sacrebleu
from transformers.trainer_pt_utils import get_parameter_names
import os 

class BaseTrainer():
    
    DEFAULT_CODET5_HYPERPARAMETERS = "configs/training/codet5.yaml"
    PEFT_CODET5_CONFIGS = "configs/training/peft_codet5.yaml"

    def __init__(self, mixed_precision=True, checkpoints_path="./checkpoints"):
        if mixed_precision:
            self.accelerator = Accelerator(mixed_precision="fp16")
        else:
            self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.checkpoints_path = checkpoints_path
        self.create_checkpoints_path(checkpoints_path)

    def get_default_codet5_hyperparameters(self):
        hyperparameters_config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS)).hyperparameters
        return hyperparameters_config
    
    def get_default_lora_config_for_codet5(self):
        
        config = OmegaConf.load(get_abs_path(self.PEFT_CODET5_CONFIGS)).lora

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False, r=config["r"], 
            lora_alpha=config["lora_alpha"], 
            lora_dropout=config["lora_dropout"]
        )
        return lora_config


    def create_checkpoints_path(self, checkpoints_path):
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
       
    def get_default_prefixtuning_config_for_codet5(self):
        
        config = OmegaConf.load(get_abs_path(self.PEFT_CODET5_CONFIGS)).prefixtuning

        prefixtuning_config = PrefixTuningConfig(
            num_virtual_tokens=config["num_virtual_tokens"],
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        return prefixtuning_config
    
    

       
    
   