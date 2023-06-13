import torch
import torch.nn as nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)


class T5RuleGenerationModel(nn.Module):
    def __init__(self, t5_model, config, tokenizer, 
                batch_size, max_source_length, 
                max_target_length, beam_size=10):

        super(T5RuleGenerationModel, self).__init__()

        self.t5_model = t5_model
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # self.max_prediction_length = 
        self.device = t5_model.device
        self.beam_size = beam_size    
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def compute_decoder_loss(self, source_ids, target_ids):
        source_mask = source_ids.ne(self.tokenizer.pad_token_id)
        target_mask = target_ids.ne(self.tokenizer.pad_token_id)
       
        outputs = self.t5_model(input_ids=source_ids, attention_mask=source_mask,
                                   labels=target_ids, decoder_attention_mask=target_mask)

        # hidden_states = outputs['decoder_hidden_states'][-1]

        return outputs.loss
    
    def generate_sequence(self, source_ids):
        source_mask = source_ids.ne(self.tokenizer.pad_token_id)
        target_ids = self.t5_model.generate(source_ids, attention_mask=source_mask, 
                            use_cache=True, num_beams=self.beam_size, 
                            early_stopping=True, max_length=self.max_target_length)
        return target_ids


    def forward(self, source_ids=None, 
                target_ids=None, generate_target=False):

        return_result = None
        if generate_target:
            target = self.generate_sequence(source_ids)
            return_result = target
        else:
            target_ids = target_ids.masked_fill(target_ids == self.tokenizer.pad_token_id, -100)
            loss = self.compute_decoder_loss(source_ids, target_ids)
            return_result = loss

        return return_result

   