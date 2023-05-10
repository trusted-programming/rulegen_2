# Instruction to generate txl rules

- Clone this project: git clone git@github.com:bdqnghi/rulegen.git.
- Install requirements: 
  ```bash
  wget https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl
  pip install torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl
  pip3 install -r requirements.txt
  SO=$(locate libtorch_python.so | grep $HOME | grep 3.8)
  export LD_LIBRARY_PATH=$(dirname $SO)
  ```
- Then, download the pretrained model from this link: https://ai4code.s3.ap-southeast-1.amazonaws.com/codet5-mising-rule-prediction.zip,
  and extract the zip file into a directory.
  ```bash
  wget https://ai4code.s3.ap-southeast-1.amazonaws.com/codet5-mising-rule-prediction.zip
  unzip codet5-mising-rule-prediction.zip
  ```
- Go to config/config.ini, change the param "pretrained_model_path" to the path of the pretrained model, e.g., 
  `codet5-base_epoch_91.bin`
- Now you are ready to use the tool, there is a folder "test_samples" that contains a few samples for testing, one can use this command to test:

  ```bash
  python3 python3 generate_missing_rule.py --before_path demo_samples/0/before.txt --after_path demo_samples//0/after.txt --context_path demo_samples/0/context.txt
  ```

It will generate the missing rule rule to the screen, sth like this:

```txl
function changeOptExpression
	replace *[opt expression]
		'--
	import RunTimeExceptionsMapper [ExceptionMapper]
	deconstruct * [exceptionTable] RunTimeExceptionsMapper
	Exception -> CSStmt [reference]
	by
		CSStmt
end function
```

Then you can check if the generated rule is aligned with the ground truth in the path ``demo_samples/1/hole_rule.txt``.

# Fine-tuning the model from Customer Trainer

```bash
python3 -m torch.distributed.run --nproc_per_node=4 test_trainer/test_t5_trainer.py
```

Please also edit the config.ini to match with your settings.

## Preliminary results To Predict the Missing Rules

### CodeT5

| Dataset    | EM    | BLEU  |
|------------|-------|-------|
| 1 Rule     | 43.56 | 74.29 |
| 2 Rule     | 34.45 | 59.25 |
| 3 Rule     | 32.57 | 43.56 |
| 4 Rule     | 21.69 | 30.15 |
| All rules  | 30.68 | 46.39 |


### CodeBERT

| Dataset    | EM    | BLEU  |
|------------|-------|-------|
| 1 Rule     | 37.21 | 70.29 |
| 2 Rule     | ----- | ----- |
| 3 Rule     | ----- | ----- |
| 4 Rule     | ----- | ----- |
| All rules  | 27.68 | 41.23 |


### StarCoder

| Dataset    | EM    | BLEU  |
|------------|-------|-------|
| 1 Rule     | ----- | ----- |
| 2 Rule     | ----- | ----- |
| 3 Rule     | ----- | ----- |
| 4 Rule     | ----- | ----- |
| All rules  | ----- | ----- |

