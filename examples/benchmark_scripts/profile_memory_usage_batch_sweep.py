import os
import subprocess
import itertools
import json
import torch

# Set environment
LOG_FILE_NAME="results/v100_batch_sweep_result.csv"

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
]

experiments = cnn_experiments + transformer_experiments + rnn_experiments
algos = ["naive", "reweight", "elegant"]
grad_sample_modes = ["hooks"]

experiments = list(itertools.product(experiments, algos, grad_sample_modes))

device_name = torch.cuda.get_device_name().split(" ")[1].split("-")[0]

for experiment in experiments:
    (model_type, architecture, input_size), algo, grad_sample_mode = experiment
    experiment = [model_type, architecture, input_size, algo, grad_sample_mode]

    # Currently, elegant doesn't support ExpandedWeights
    if algo == "elegant" and grad_sample_mode == "ew":
        continue

    if algo == "sgd":
        dpsgd_mode = "--disable-dp"
    else:
        dpsgd_mode = f"--dpsgd_mode {algo}"

    # Find max batch size
    if model_type == "cnn":
        batch_size = 16
    else:
        batch_size = 4
    return_code = 0
    while return_code == 0:
        args = f"nice -n 10 taskset -c 0-20 python benchmark.py --steps 20 --input_size {input_size} --architecture {architecture} --model_type {model_type} {dpsgd_mode} --grad_sample_mode {grad_sample_mode} --batch_size {batch_size} --profile_time --log_file {LOG_FILE_NAME}"
        ret = subprocess.run(args, shell=True)
        return_code = ret.returncode
        if return_code != 0:
            break
        batch_size *= 2