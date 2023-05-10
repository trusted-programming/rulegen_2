

import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from accelerate import Accelerator,DistributedType
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,logging,set_seed, get_linear_schedule_with_warmup, AdamW
from omegaconf import OmegaConf
from accelerate import Accelerator
from codetf.trainer.base_trainer import BaseTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig
from codetf.common.utils import get_abs_path
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sacrebleu
import os 

class CodeT5Seq2SeqTrainer(BaseTrainer):    
    def __init__(self, train_dataset, validation_dataset=None,
                tokenizer=None, checkpoints_path="./checkpoints", mixed_precision=False,
                peft=None, pretrained_model_path="Salesforce/codet5-base-multi-sum", training_args=None, evaluator=None, evaluation_fn=None):
        
        super().__init__(mixed_precision, checkpoints_path)

        
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.model.to(self.device)
        self.tokenizer = tokenizer

        if peft:
            if peft == "lora":
                peft_config = self.get_default_lora_config_for_codet5()
            # elif peft == "adalora":
            #     peft_config = self.get_default_adalora_config_for_codet5()
            elif peft == "prefixtuning":
                peft_config = self.get_default_prefixtuning_config_for_codet5()
                
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        if training_args == None:
            self.training_args = self.get_default_codet5_hyperparameters()
        else:
            self.training_args = training_args

        self.evaluation_fn = evaluation_fn
        self.evaluator = evaluator
        self.train_dataset = train_dataset
        if validation_dataset is not None:
            self.validation_dataset = validation_dataset
            self.validation_dataloader = DataLoader(self.validation_dataset, shuffle=False, batch_size=self.training_args["batch_size"], num_workers=self.training_args["num_workers"])
            self.validation_dataloader = self.accelerator.prepare(self.validation_dataloader)
        else:
            self.validation_dataset = None
            self.validation_dataloader = None

        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.training_args["batch_size"], num_workers=self.training_args["num_workers"])

        self.model, self.train_dataloader = self.accelerator.prepare(
            self.model, self.train_dataloader
        )

        self.set_up_training_params()
        

    def set_up_training_params(self):
        num_train_optimization_steps = self.training_args["num_train_epochs"] * len(self.train_dataloader)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args["learning_rate"], eps=self.training_args["adam_epsilon"])

        if self.training_args["warmup_steps"] < 1:
            warmup_steps = num_train_optimization_steps * self.training_args["warmup_steps"]
        else:
            warmup_steps = self.training_args["warmup_steps"]

        self.lr_scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.training_args["learning_rate"], 
                                epochs=self.training_args["num_train_epochs"], steps_per_epoch=len(self.train_dataloader))

    def generate_text(self, source_ids):
        source_mask = source_ids.ne(self.tokenizer.pad_token_id)
        output_ids = self.model.module.generate(input_ids=source_ids, attention_mask=source_mask, 
                            use_cache=True, num_beams=self.training_args["beam_size"], 
                            early_stopping=True, max_length=self.training_args["max_prediction_length"])

        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    def evaluate(self):
        if self.validation_dataloader is None:
            raise ValueError("Validation dataset was not provided during the Trainer's initialization. Please provide a validation dataset to perform evaluation.")

        self.model.eval()

        hypotheses = []
        references = []

        eval_dataloader = tqdm(self.validation_dataloader, desc="Evaluating", unit="batch")
        for batch in eval_dataloader:
            source_ids, target_ids = batch

            with torch.no_grad():
                decoded_generated = self.generate_text(source_ids)

            if not decoded_generated:
                print(f"Empty decoded_generated for source_ids: {source_ids}")

            decoded_target = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
            hypotheses.extend(decoded_generated)
            references.extend(decoded_target)

        if self.evaluation_fn is not None:
            score = self.evaluation_fn(hypotheses, references)
        elif self.evaluator is not None:
            score = self.evaluator.compute(hypotheses, references)

    def evaluate(self):
        self.model.eval()
        hypotheses = []
        references = []


        validation_dataloader = tqdm(self.validation_dataloader, desc="Evaluating", unit="batch")
        for batch in validation_dataloader:
            source_ids, target_ids = batch

            with torch.no_grad():
                decoded_generated = self.generate_text(source_ids)

            if not decoded_generated:
                print(f"Empty decoded_generated for source_ids: {source_ids}")

            decoded_target = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
            hypotheses.extend(decoded_generated)
            references.extend(decoded_target)

        if self.evaluation_fn is not None:
            score = self.evaluation_fn(hypotheses, references)
        elif self.evaluator is not None:
            score = self.evaluator.compute(hypotheses, references)
        else:
            raise ValueError("No evaluation function or Evaluator instance provided. Please provide one of them")

        return score

    def train(self):
        num_epochs = self.training_args["num_train_epochs"]
        best_score = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_dataloader = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    source_ids, target_ids = batch
                    source_mask = source_ids.ne(self.tokenizer.pad_token_id)
                    target_mask = target_ids.ne(self.tokenizer.pad_token_id)

                    with self.accelerator.autocast():
                        outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask)
                        loss = outputs.loss

                    self.accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args["max_grad_norm"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    train_dataloader.set_postfix({"Loss": loss.item()})

            score = self.evaluate()
            self.accelerator.print(f"Score for epoch {epoch+1}: {score:.2f}")
            if score > best_score:
                best_score = score
                self.accelerator.save(self.model.state_dict(), f"{self.checkpoints_path}/best_model.pt")
                self.accelerator.print(f"New best model saved with score: {best_score:.2f}")
     