from flask import Flask, request, jsonify
import subprocess
import os
import torch
import torch.nn as nn
from model.t5_rule_generation_model import T5RuleGenerationModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import configparser
import logging
from accelerate import Accelerator
from collections import OrderedDict

accelerator = Accelerator()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

config_ini_path = "config/t5_config.ini"
configs = configparser.ConfigParser()
configs.read(config_ini_path)

config = configs["neural_network"]

checkpoint_path = config["checkpoint_path"]

existing_model_checkpoint = config["pretrained_model_path"]

config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
encoder_config = config_class.from_pretrained(config["config_path"])
pretrained_model = model_class.from_pretrained(config["model_name"], device_map="auto")
tokenizer = tokenizer_class.from_pretrained(config["tokenizer_name"])

model = T5RuleGenerationModel(t5_model=pretrained_model, config=encoder_config, tokenizer=tokenizer, 
                            batch_size=config.getint("batch_size"), 
                            max_source_length=config.getint("max_source_length"), 
                            max_target_length=config.getint("max_target_length"), 
                            beam_size=config.getint("beam_size"))
if os.path.exists(existing_model_checkpoint):
    print("Resume checkpoint")
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

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    before = data.get('before')
    after = data.get('after')
    context = data.get('context')
    

    if not all([before, after, context]):
        return jsonify({'error': 'Invalid inputs'}), 400

    missing_rule = generate_missing_rule(before, after, context)
    return jsonify({'missing_rule': missing_rule}), 200


@app.route('/execute', methods=['POST'])
def execute():
    data = request.get_json()
    program = data.get('program')
    txl = data.get('txl')

    if not all([program, txl]):
        return jsonify({'error': 'Invalid inputs'}), 400

    # Save the input program to a temporary file
    with open('temp.txt', 'w') as f:
        f.write(program)

    with open('temp_txl.txl', 'w') as f:
        f.write(txl)

    try:
        # Call TXL and capture the output
        output = subprocess.check_output(['txl', 'temp.txt', 'temp_txl.txl'])

        # Delete the temporary file
        os.remove('temp.txt')
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'TXL execution failed'}), 500

    return jsonify({'transformed_program': output.decode('utf-8')}), 200

def generate_missing_rule(before, after, context):
   
    # read the contents from string instead of files
    # with open(before_path, "r") as f1:
    before_content = before

    # with open(after_path, "r") as f2:
    after_content = after

    # with open(context_path, "r") as f3:
    context_content = context

    source_input = f"{before_content} <unk> {after_content} <unk> {context_content}"
    source_input_ids = tokenizer.encode(source_input, max_length=config.getint("max_source_length"), padding='max_length', truncation=True)
    source_input_ids = torch.tensor([source_input_ids])
    source_input_ids = source_input_ids.to(model.device)

    predict_ids = model(source_ids=source_input_ids, generate_target=True)
    predict_nl = tokenizer.decode(predict_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Return the prediction instead of printing it
    return predict_nl


if __name__ == "__main__":
    

    
    app.run(host='0.0.0.0', port=5000)