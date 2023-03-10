class GPTRuleGenerationModel(nn.Module):
    def __init__(self, gpt2_model, config, tokenizer, 
                batch_size, max_source_length, 
                max_target_length, beam_size=10):

        super(GPTRuleGenerationModel, self).__init__()

        self.gpt_model = gpt_model
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.beam_size = beam_size
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def compute_decoder_loss(self, rust_input_ids, rule_output_ids):
        rust_input_mask = rust_input_ids.ne(self.tokenizer.pad_token_id)
        rule_output_mask = rule_output_ids.ne(self.tokenizer.pad_token_id)
       
        # Here we pass the input and target as the only two inputs 
        # to the model, and the labels are the target
        outputs = self.gpt_model(input_ids=rust_input_ids, attention_mask=rust_input_mask, 
                                   labels=rule_output_ids)

        # The loss is the output of the model for the input and target
        return outputs[0]
    
    def generate_sequence(self, rust_input_ids):
        rust_input_mask = rust_input_ids.ne(self.tokenizer.pad_token_id)
        rule_ids = self.gpt2_model.generate(input_ids=rust_input_ids, attention_mask=rust_input_mask, 
                            use_cache=True, num_beams=self.beam_size, 
                            early_stopping=True, max_length=self.max_target_length)
        return rule_ids

    def forward(self, rust_input_ids=None, 
                rule_output_ids=None, generate_txl=False):

        return_result = None
        if generate_txl:
            txl = self.generate_sequence(rust_input_ids)
            return_result = txl
        else:
            rule_output_ids = rule_output_ids.masked_fill(rule_output_ids == self.tokenizer.pad_token_id, -100)
            loss = self.compute_decoder_loss(rust_input_ids, rule_output_ids)
            return_result = loss

        return return_result