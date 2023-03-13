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
- Then, download the pretrained model from this link: https://ai4code.s3.ap-southeast-1.amazonaws.com/codet5-rulegen.zip,
  and extract the zip file into a directory.
  ```bash
  wget https://ai4code.s3.ap-southeast-1.amazonaws.com/codet5-rulegen.zip
  unzip codet5-rulegen.zip
  ```
- Go to config/config.ini, change the param "pretrained_rulegen_model_path" to the path of the pretrained model, e.g., 
  `codet5-base_epoch_72.bin`
- Now you are ready to use the tool, there is a folder "test_samples" that contains a few samples for testing, one can use this command to test:

  ```bash
  python3 generate_txl.py --source_path test_samples/1/source.txt --target_path test_samples/1/target.txt
  ```

It will generate the txl rule to the screen, sth like this:

```txl
rule removeExplicitBlocks replace [expression] { B [expression] } ; R [expression] by B [appendExpression R] end rule
```

Then you can check if the generated rule is aligned with the ground truth in the path ``test_samples/1/ground_truth.txt``.

# Train the model from scratch

```bash
python3 -m torch.distributed.run --nproc_per_node=4 main.py
```

Please also edit the config.ini to match with your settings.
