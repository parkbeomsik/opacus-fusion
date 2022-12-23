import json
import itertools
import subprocess

# Set environment
CPU_AFFINITY=0-2
LOG_FILE_NAME="results/v100_memory_result.csv"
MAX_BATCH_SIZE_FILE_NAME="results/V100_max_batch_size.json"

# Batch size configuration
with open(MAX_BATCH_SIZE_FILE_NAME, "r") as f:
    batch_size_dict = json.load(f)

cnn_experiments = [
    ("cnn", "resnet18", "32"), 
    ("cnn", "resnet50", "32"), 
    ("cnn", "resnet152", "32"),
    # ("cnn", "resnet18", "224"),
    # ("cnn", "resnet50", "224"),
    # ("cnn", "resnet152", "224")
]

transformer_experiments = [
    ("transformer", "bert-base", "32"),
    ("transformer", "bert-large", "32"), 
    # ("transformer", "bert-base", "64"),
    # ("transformer", "bert-large", "64"), 
    # ("transformer", "bert-base", "128"), 
    # ("transformer", "bert-large", "128"),
    # ("transformer", "bert-base", "256"), 
    # ("transformer", "bert-large", "256"),
    # ("transformer", "bert-base", "512"),
    # ("transformer", "bert-large", "512")
]

rnn_experiments = [    
    # ("rnn", "deepspeech", "32"), 
    # ("rnn", "gnmt", "32"),
    # ("rnn", "deepspeech", "64"), 
    ("rnn", "gnmt", "64"),
    # ("rnn", "deepspeech", "128"), 
    # ("rnn", "gnmt", "128"),
    # ("rnn", "deepspeech", "256"), 
    # ("rnn", "gnmt", "256"),
    ("rnn", "deepspeech", "512"),
    # ("rnn", "gnmt", "512"),
    # ("rnn", "deepspeech", "4096"),
]

experiments = cnn_experiments + transformer_experiments + rnn_experiments
algos = ["naive", "reweight", "elegant"]

batch_mults = [1, 2, 4, 8, 16, 32]

# experiments = list(itertools.product(experiments, algos))

for case in experiments:
    for batch_mult in batch_mults:
        for algo in algos:
            print(case, algo, batch_mult)
            (model_type, architecture, input_size) = case
            quant = False
            if algo == "elegant-quant":
                quant = True
                algo = "elegant"

            if algo == "sgd":
                dpsgd_mode = "--disable-dp"
            else:
                dpsgd_mode = f"--dpsgd_mode {algo}"

            batch_size = batch_size_dict[" ".join((model_type, architecture, input_size, "naive", "hooks"))] // 4

            args = f"nice -n 10 taskset -c 0-20 python benchmark.py --warm_up_steps 0 --steps 1 --input_size {input_size} --architecture {architecture} --model_type {model_type} {dpsgd_mode} {'--quant' if quant else ''} --batch_size {batch_size*batch_mult} --profile_memory --log_file {LOG_FILE_NAME} --adaptive_clipping"
            print(args)
            ret = subprocess.run(args, shell=True)
            return_code = ret.returncode
            if return_code != 0:
                print("Failed!!")
                print(args)
                # exit(0)