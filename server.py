from flask import Flask, request, jsonify
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    before = data.get('before')
    after = data.get('after')
    context = data.get('context')

    if not all([before, after, context]):
        return jsonify({'error': 'Invalid inputs'}), 400

    # missing_rule = generate_missing_rule(before, after, context)
    return jsonify({'missing_rule': "aaa"}), 200

# def generate_missing_rule(before, after, context):
   
#     # local_rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # config_ini_path = "config/t5_config.ini"
#     # configs = configparser.ConfigParser()
#     # configs.read(config_ini_path)

#     # config = configs["neural_network"]

#     # checkpoint_path = config["checkpoint_path"]
#     # checkpoint_name = config["checkpoint_name"]

#     # config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
#     # encoder_config = config_class.from_pretrained(config["config_path"])
#     # pretrained_model = model_class.from_pretrained(config["model_name"])
#     # tokenizer = tokenizer_class.from_pretrained(config["tokenizer_name"])


#     # model = T5RuleGenerationModel(t5_model=pretrained_model, config=encoder_config, tokenizer=tokenizer, 
#     #                             batch_size=config.getint("batch_size"), 
#     #                             max_source_length=config.getint("max_source_length"), 
#     #                             max_target_length=config.getint("max_target_length"), 
#     #                             beam_size=config.getint("beam_size"))
    
#     # model.to(local_rank)

#     # read the contents from string instead of files
#     # with open(before_path, "r") as f1:
#     before_content = before

#     # with open(after_path, "r") as f2:
#     after_content = after

#     # with open(context_path, "r") as f3:
#     context_content = context

#     source_input = f"{before_content} <unk> {after_content} <unk> {context_content}"
#     source_input_ids = tokenizer.encode(source_input, max_length=config.getint("max_source_length"), padding='max_length', truncation=True)
#     source_input_ids = torch.tensor([source_input_ids])
#     source_input_ids = source_input_ids.to(local_rank)

#     predict_ids = model(source_ids=source_input_ids, generate_target=True)
#     predict_nl = tokenizer.decode(predict_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

#     # Return the prediction instead of printing it
#     return predict_nl


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)