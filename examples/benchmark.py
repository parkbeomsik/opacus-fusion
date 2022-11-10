#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs CIFAR10 training with differential privacy.
"""

import argparse
from functools import partial
import time
import gc
import os
from inspect import getframeinfo, stack

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from torchvision import models
from tqdm import tqdm

from transformers import BertConfig, BertForSequenceClassification
from models.deepspeech import DeepSpeech
from models.gnmt.gnmt import GNMT
from models.resnet import resnet18, resnet50, resnet152

from opacus.profiler import profiler, total_ignored_time
from opacus import config

def debuginfo(message=None):
    caller = getframeinfo(stack()[2][0])
    print("%s:%d" % (caller.filename, caller.lineno))

def print_all_tensors():
    # Print all tensors
    print("\n==========================================================================================================")
    caller = getframeinfo(stack()[1][0])
    print("%s:%d" % (caller.filename, caller.lineno))
    gc.collect()
    all_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(id(obj), type(obj), obj.size(), obj.device)
        except:
            pass

    # all_tensors.sort(key=lambda x: x[0])
    # for tensor in all_tensors:
    #     print(tensor)

def pretty_number(n):
    if n >= 1e6:
        return f"{n / 1e6: .2f}M"
    elif n >= 1e3:
        return f"{n / 1e3: .2f}K"
    else:
        return str(n)

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, bias=False)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1000)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], x.shape[1])  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

def print_args(args):
    print("==================================================================")
    print("                          Configuration                           ")
    print("==================================================================")
    print(f"DPSGD mode       : {args.dpsgd_mode}")
    print(f"Quantization     : {args.quant}")
    print(f"Model type       : {args.model_type}")
    print(f"Architecture     : {args.architecture}")
    print(f"Input size       : {args.input_size}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Profile time     : {args.profile_time}")
    print(f"Verbose          : {args.verbose}")

def main():  # noqa: C901
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    world_size = 1

    args = parse_args()

    print_args(args)

    if args.dpsgd_mode == "naive":
        config.dpsgd_mode = config.MODE_NAIVE
    elif args.dpsgd_mode == "reweight":
        config.dpsgd_mode = config.MODE_REWEIGHT
    elif args.dpsgd_mode == "elegant":
        config.dpsgd_mode = config.MODE_ELEGANT

    config.grad_sample_mode = args.grad_sample_mode
    config.quantization = args.quant

    config.profile_value = args.profile_value
    if config.profile_value:
        args.warm_up_steps = 0
        args.steps = 1
        if (args.model_load_path is None
            or args.input_load_path is None
            or args.grad_save_path is None):
            assert 0, "Config is wrong."

    config.profile_throughput = args.profile_throughput
    config.profile_time = args.profile_time # or args.profile_throughput
    config.profile_memory = args.profile_memory
    config.verbose = args.verbose

    config.model_type = args.model_type
    config.architecture = args.architecture
    config.batch_size = args.batch_size

    config.grad_save_path = args.grad_save_path

    B = args.batch_size

    if args.model_type == "cnn":
        if args.architecture == "sample_conv_net":
            model = SampleConvNet()
        elif args.architecture == "resnet18":
            model = resnet18(pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c)))
        elif args.architecture == "resnet50":
            model = resnet50(pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c)))
        elif args.architecture == "resnet152":
            model = resnet152(pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c)))
        else:
            model = models.__dict__[args.architecture](
                pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c))
            )

        model = model.to(memory_format=torch.channels_last)

        # Save or load model
        if args.model_load_path:
            model.load_state_dict(torch.load(args.model_load_path))
        if args.model_save_path:
            torch.save(model.state_dict(), args.model_save_path)

        inputs = torch.randn(1, 3, args.input_size, args.input_size)
        inputs = inputs.to(memory_format=torch.channels_last)

        # Save of load inputs
        if args.input_load_path:
            inputs = torch.load(args.input_load_path)
        if args.input_save_path:
            torch.save(inputs, args.input_save_path)

        inputs = inputs.expand(B, 3, args.input_size, args.input_size)
        inputs = inputs.expand(
            args.warm_up_steps + args.steps, B, 3, args.input_size, args.input_size
            ).reshape((args.warm_up_steps + args.steps) * B, 3, args.input_size, args.input_size)
        print(inputs.stride())
        labels = torch.ones(B, dtype=torch.int64).repeat(args.warm_up_steps + args.steps)
        print(inputs.sum())

        train_dataset = TensorDataset(inputs, labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=B,
            num_workers=args.workers,
            pin_memory=True,
        )

    if args.model_type == "transformer":
        input_ids = torch.randint(0, 20000, (B, args.input_size)).type(torch.long)
        attention_mask = torch.randint(1, 2, (B, args.input_size)).type(torch.long)
        token_type_ids = torch.randint(0, 1, (B, args.input_size)).type(torch.long)

        input_ids = input_ids.expand(
                    args.warm_up_steps + args.steps, B, args.input_size
                    ).reshape((args.warm_up_steps + args.steps) * B, args.input_size)
        attention_mask = attention_mask.expand(
                    args.warm_up_steps + args.steps, B, args.input_size
                    ).reshape((args.warm_up_steps + args.steps) * B, args.input_size)
        token_type_ids = token_type_ids.expand(
                    args.warm_up_steps + args.steps, B, args.input_size
                    ).reshape((args.warm_up_steps + args.steps) * B, args.input_size)
        
        inputs = (input_ids, attention_mask, token_type_ids)
        labels = torch.ones(B, dtype=torch.int64).repeat(args.warm_up_steps + args.steps)

        bert_config = BertConfig.from_pretrained(
            args.architecture + "-cased",
            num_labels=3,
        )
        model = BertForSequenceClassification.from_pretrained(
            args.architecture + "-cased",
            config=bert_config,
        )

        model.bert.embeddings.position_embeddings.requires_grad_(False)
        model.bert.embeddings.token_type_embeddings.requires_grad_(False)

        train_dataset = TensorDataset(*inputs, labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=B,
            num_workers=args.workers,
            pin_memory=True,
        )

    if args.model_type == "rnn":
        if args.architecture == "deepspeech":
            inputs = torch.randn(B, 161, args.input_size)
            labels = torch.ones(B, 10)

            inputs = inputs.expand(
                    args.warm_up_steps + args.steps, B, 161, args.input_size
                    ).reshape((args.warm_up_steps + args.steps) * B, 161, args.input_size)
            labels = torch.ones(B, 10, dtype=torch.int64).repeat(args.warm_up_steps + args.steps, 1)
            # label_lengths = torch.Tensor([10 for _ in range(B)]).repeat(args.warm_up_steps + args.steps)

            model = DeepSpeech(bidirectional=False, 
                               labels = ['_', "'", 
                                         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                                         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                                         'U', 'V', 'W', 'X', 'Y', 'Z', ' '],)

            train_dataset = TensorDataset(inputs, labels)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=B,
                num_workers=args.workers,
                pin_memory=True,
            )

        if args.architecture == "gnmt":
            input_ids = torch.randint(0, 20000, (B, args.input_size)).type(torch.long)
            input_lengths = torch.Tensor([args.input_size]*B).type(torch.long)
            decoder_input_ids = torch.randint(0, 20000, (B, args.input_size)).type(torch.long)

            input_ids = input_ids.expand(
                        args.warm_up_steps + args.steps, B, args.input_size
                        ).reshape((args.warm_up_steps + args.steps) * B, args.input_size)
            input_lengths = input_lengths.expand(
                        args.warm_up_steps + args.steps, B
                        ).reshape((args.warm_up_steps + args.steps) * B)
            decoder_input_ids = decoder_input_ids.expand(
                        args.warm_up_steps + args.steps, B, args.input_size
                        ).reshape((args.warm_up_steps + args.steps) * B, args.input_size)
            
            inputs = (input_ids, input_lengths, decoder_input_ids)
            labels = torch.zeros(B, 32000, dtype=torch.int64)
            labels[:, 1] = torch.ones(B)
            labels = labels.repeat(args.warm_up_steps + args.steps, 1)

            model = GNMT()

            train_dataset = TensorDataset(*inputs, labels)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=B,
                num_workers=args.workers,
                pin_memory=True,
            )

    # model = SampleConvNet()

    model.train()
    model = model.to(args.device)
    print("Model size: " + pretty_number(sum([p.numel() for p in model.parameters()])))

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    privacy_engine = None
    if not args.disable_dp:
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                args.max_per_sample_grad_norm / np.sqrt(n_layers)
            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm

        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_mode,
        )
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
            loss_reduction="mean",
            grad_sample_mode=config.grad_sample_mode
        )

    if args.architecture == "deepspeech":
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        def criterion_func(output, target):
            return criterion(output[0], target, output[1], 
                             torch.Tensor([target.shape[1] for _ in range(target.shape[0])]).to(torch.int))
        criterion_non_reduction = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
        def criterion_func_non_reduction(output, target):
            return criterion_non_reduction(output[0], target, output[1], 
                                           torch.Tensor([target.shape[1] for _ in range(target.shape[0])]).to(torch.int))
    else:
        criterion_func = nn.CrossEntropyLoss(reduction="mean")
        criterion_func_non_reduction = nn.CrossEntropyLoss(reduction="mean")

    model.train()
    print(model)

    if args.benchmark_data_loader:
        torch.cuda.synchronize()
        start = time.time()
        profiler.init_step()
        for i, batch in tqdm(enumerate(train_loader)):
            inputs = batch[0:-1]
            target = batch[-1]

            if i == args.warm_up_steps:
                start = time.time()

            for tensor_idx, tensor in enumerate(inputs):
                if config.model_type == "cnn":
                    inputs[tensor_idx] = tensor.cuda(non_blocking=True).to(memory_format=torch.channels_last)
                else:
                    inputs[tensor_idx] = tensor.cuda(non_blocking=True)
            
            print(f"inputs stride : {inputs[0].stride()}")
            profiler.record("Data loading")

            # compute output
            optimizer.zero_grad()
            if args.model_type == "cnn" or args.model_type == "rnn":
                output = model(*inputs)
            if args.model_type == "transformer":
                output = model(*inputs, return_dict=False)[0]
            loss = criterion_func(output, target.cuda(non_blocking=True))
            profiler.record("Forward", input_activation=True)

            loss.backward()
            profiler.record("Backward activation")

            if args.disable_dp:
                optimizer.step()
            else:
                optimizer.step(input = inputs, criterion = criterion_func_non_reduction, target=target.cuda(non_blocking=True))
            profiler.record("Update")
            profiler.end_step()

            print(f"Peak memory usage = {torch.cuda.max_memory_allocated()}")

            if i == args.warm_up_steps + 1:
                profiler.__init__()

            if i == args.warm_up_steps + args.steps - 1:
                torch.cuda.synchronize()
                end = time.time()

    else:
        images = torch.randn(B, 3, H, W).cuda()
        target = torch.arange(B).cuda()

        torch.cuda.synchronize()
        start = time.time()
        for _ in tqdm(range(args.warm_up_steps + args.steps)):
            optimizer.zero_grad(True)
            # print_all_tensors()
            output = model(images)
            loss = criterion(output, target)
            print(torch.cuda.memory_allocated())
            loss.backward()
            # print_all_tensors()
            optimizer.step(input = images, criterion = nn.CrossEntropyLoss(reduction="none"), target=target)
            # print_all_tensors()

    torch.cuda.synchronize()

    print(f"Peak memory usage = {torch.cuda.max_memory_allocated()}")
    
    if config.profile_throughput:
        print("")
        print("==============================================================================================")
        print("")
        print("                                     Throughput (#examples/s)")
        print("")
        print((args.steps * args.batch_size / (end - start - sum(total_ignored_time[args.warm_up_steps:]))))

        throughput = args.steps * args.batch_size / (end - start - sum(total_ignored_time[args.warm_up_steps:]))

        if args.log_file != "":
            if os.path.exists(args.log_file):
                with open(args.log_file, "a") as f:
                    f.write(f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}_{args.grad_sample_mode}_{'int8' if args.quant else 'no'},{throughput}\n")
            else:
                with open(args.log_file, "w") as f:
                    f.write("Config,Throughput (#example/s)\n")
                    f.write(f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}_{args.grad_sample_mode}_{'int8' if args.quant else 'no'},{throughput}\n")

    if config.profile_time and not config.profile_throughput:
        print("==============================================================================================")
        print("")
        print("                                     Time records (ms)")
        print("")
        print(profiler.time_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}"]))
        print("")
        print(profiler.time_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}"]).to_csv())

        if args.log_file != "":
            if os.path.exists(args.log_file):
                with open(args.log_file, "a") as f:
                    row = profiler.time_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}_{args.grad_sample_mode}_{'int8' if args.quant else 'no'}"]).to_csv()
                    f.write(row.split('\n')[1] + "\n")
            else:
                with open(args.log_file, "w") as f:
                    row = profiler.time_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}_{args.grad_sample_mode}_{'int8' if args.quant else 'no'}"]).to_csv()
                    f.write(row)

    if config.profile_memory:
        print("")
        print("==============================================================================================")
        print("")
        print("                                     Memory records")
        print("")
        print(profiler.memory_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}"]))  
        print("")
        print(profiler.memory_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}"]).to_csv())

        if args.log_file != "":
            if os.path.exists(args.log_file):
                with open(args.log_file, "a") as f:
                    row = profiler.memory_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}_{args.grad_sample_mode}_{'int8' if args.quant else 'no'}"]).to_csv()
                    f.write(row.split('\n')[1] + "\n")
            else:
                with open(args.log_file, "w") as f:
                    row = profiler.memory_as_df([f"{args.architecture}_{args.input_size}x{args.input_size}_{args.batch_size}_{args.dpsgd_mode}_{args.grad_sample_mode}_{'int8' if args.quant else 'no'}"]).to_csv()
                    f.write(row)

def parse_args():
    parser = argparse.ArgumentParser(description="Opacus Imagenet Benchmark")
    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--steps",
        default=100,
        type=int,
        help="Number of steps",
    )
    parser.add_argument(
        "--benchmark-data-loader",
        action="store_true",
        default=True,
        help="Also benchmark data loader",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset, this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--input_size",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )

    parser.add_argument(
        "--gn-groups",
        type=int,
        default=8,
        help="Number of groups in GroupNorm",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-mode",
        action="store_true",
        default=False,
        help="Enable Secure mode to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="path to save check points",
    )

    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to run the code."
    )

    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=['cnn', 'transformer', 'rnn'],
        help="Model type. (cnn, transformer, rnn)"
    )

    parser.add_argument(
        "--quant",
        action="store_true",
        default=False,
        help="INT8 quantization.",
    )

    parser.add_argument(
        "--profile_time",
        action="store_true",
        default=False,
        help="Profile time.",
    )

    parser.add_argument(
        "--profile_throughput",
        action="store_true",
        default=False,
        help="Profile throughput.",
    )

    parser.add_argument(
        "--profile_memory",
        action="store_true",
        default=False,
        help="Profile memory.",
    )
    
    parser.add_argument(
        "--profile_value",
        action="store_true",
        default=False,
        help="Profile value.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose.",
    )

    parser.add_argument(
        "--warm_up_steps",
        default=10,
        type=int,
        help="Number of warm-up steps",
    )

    parser.add_argument(
        "--dpsgd_mode", type=str, default="naive", help="DPSGD mode. (naive, reweight, elegant)"
    )

    parser.add_argument(
        "--grad_sample_mode", type=str, default="hooks", help="Grad sample mode. (hooks, ew)"
    )

    parser.add_argument(
        "--log_file", type=str, default="", help="logging file name."
    )

    parser.add_argument(
        "--model_load_path", type=str, default=None, help="model path to load."
    )

    parser.add_argument(
        "--model_save_path", type=str, default=None, help="model path to save"
    )

    parser.add_argument(
        "--input_load_path", type=str, default=None, help="input path to load."
    )

    parser.add_argument(
        "--input_save_path", type=str, default=None, help="input path to save"
    )

    parser.add_argument(
        "--grad_save_path", type=str, default=None, help="grad path to save"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
