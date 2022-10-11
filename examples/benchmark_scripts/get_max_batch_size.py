import os
import subprocess
import json

cnn_experiments = [
    ("cnn", "resnet18", "32"), ("cnn", "resnet18", "224"),
    ("cnn", "resnet50", "32"), ("cnn", "resnet50", "224"),
    ("cnn", "resnet152", "32"), ("cnn", "resnet152", "224")
]

transformer_experiments = [
    ("transformer", "bert-base", "32"), ("transformer", "bert-base", "64"), ("transformer", "bert-base", "128"), ("transformer", "bert-base", "256"),
    ("transformer", "bert-large", "32"), ("transformer", "bert-large", "64"), ("transformer", "bert-large", "128"), ("transformer", "bert-large", "256")
]

rnn_experiments = [
    ("rnn", "gnmt", "32"), ("rnn", "gnmt", "64"), ("rnn", "gnmt", "128"), ("rnn", "gnmt", "256"),
    ("rnn", "deepspeech", "32"), ("rnn", "deepspeech", "64"), ("rnn", "deepspeech", "128"), ("rnn", "deepspeech", "256"),
]

experiments = cnn_experiments + transformer_experiments + rnn_experiments

if os.path.exists("results/max_batch_size.json"):
    with open("results/max_batch_size.json", "r") as f:
        batch_size_dict = json.load(f)
else:
    batch_size_dict = {}

for experiment in experiments:
    model_type, architecture, input_size = experiment
    batch_size = 1
    return_code = 0
    while return_code == 0:
        args = f"nice -n 10 taskset -c 0-20 python benchmark.py --steps 10 --input_size {input_size} --architecture {architecture} --model_type {model_type} --dpsgd_mode naive --batch_size {batch_size}"
        ret = subprocess.run(args, shell=True)
        return_code = ret.returncode
        if return_code != 0:
            break
        batch_size *= 2

    batch_size_dict[" ".join(experiment)] = batch_size

    with open("results/max_batch_size.json", "w") as f:
        json.dump(batch_size_dict, f, indent=4)