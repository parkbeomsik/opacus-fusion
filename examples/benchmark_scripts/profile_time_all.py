import json
import itertools
import subprocess

# Set environment
CPU_AFFINITY=0-2
LOG_FILE_NAME="results/a100_result.csv"
MAX_BATCH_SIZE_FILE_NAME="results/A100_max_batch_size.json"

# Batch size configuration
with open(MAX_BATCH_SIZE_FILE_NAME, "r") as f:
    batch_size_dict = json.load(f)

cnn_experiments = [
    ("cnn", "resnet18", "32"), ("cnn", "resnet18", "224"),
    ("cnn", "resnet50", "32"), ("cnn", "resnet50", "224"),
    ("cnn", "resnet152", "32"), ("cnn", "resnet152", "224")
]

transformer_experiments = [
    ("transformer", "bert-base", "32"), ("transformer", "bert-base", "64"), ("transformer", "bert-base", "128"), ("transformer", "bert-base", "256"), ("transformer", "bert-base", "512"),
    ("transformer", "bert-large", "32"), ("transformer", "bert-large", "64"), ("transformer", "bert-large", "128"), ("transformer", "bert-large", "256"), ("transformer", "bert-large", "512")
]

rnn_experiments = [
    ("rnn", "gnmt", "32"), ("rnn", "gnmt", "64"), ("rnn", "gnmt", "128"), ("rnn", "gnmt", "256"), ("rnn", "gnmt", "512"),
    ("rnn", "deepspeech", "32"), ("rnn", "deepspeech", "64"), ("rnn", "deepspeech", "128"), ("rnn", "deepspeech", "256"), ("rnn", "deepspeech", "512"),
]

experiments = transformer_experiments + rnn_experiments
algos = ["naive", "reweight", "elegant", "elegant-quant"]

experiments = list(itertools.product(experiments, algos))

for case in experiments:
    print(case)
    (model_type, architecture, input_size), algo = case
    quant = False
    if algo == "elegant-quant":
        quant = True
        algo = "elegant"

    batch_size = batch_size_dict[" ".join((model_type, architecture, input_size))]

    args = f"nice -n 10 taskset -c 0-20 python benchmark.py --steps 100 --input_size {input_size} --architecture {architecture} --model_type {model_type} --dpsgd_mode {algo} {'--quant' if quant else ''} --batch_size {batch_size} --profile_time --log_file {LOG_FILE_NAME}"
    print(args)
    ret = subprocess.run(args, shell=True)
    return_code = ret.returncode
    if return_code != 0:
        print("Failed!!")
        print(args)
        exit(0)