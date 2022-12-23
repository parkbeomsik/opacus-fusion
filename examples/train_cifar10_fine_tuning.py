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
import logging
from functools import partial
import time
import gc
import os
import sys
from datetime import datetime, timedelta
from inspect import getframeinfo, stack
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from torchvision import models
from tqdm import tqdm

from torchvision.datasets import CIFAR10
from models.resnet_jax import resnet18, resnet50, resnet152
from models.convert_objax_to_pytorch import load_resnet18_from_objax

from opacus.profiler import profiler, total_ignored_time
from opacus.profiler import start_timer, pause_timer, stop_timer, get_elapsed_time
from opacus import config

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ddp")
logger.setLevel(level=logging.INFO)

best_acc1 = 0

def str_args(args):
    ret = ""
    ret += "==================================================================\n"
    ret += "                          Configuration                           \n"
    ret += "==================================================================\n"
    # ret += f"DPSGD mode       : {args.dpsgd_mode}\n"
    # ret += f"Model type       : cnn\n"
    # ret += f"Architecture     : {args.architecture}\n"
    # ret += f"Batch size       : {args.batch_size}\n"
    for k, v in vars(args).items():
        ret += f"{k:<30}: {v}\n"

    return(ret)

def accuracy(preds, labels):
    return (preds == labels).mean()

def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
    start_time = datetime.now()

    model.train()
    criterion = nn.CrossEntropyLoss()
    criterion_func_non_reduction = nn.CrossEntropyLoss(reduction="none")

    losses = []
    top1_acc = []

    if args.physical_batch_size == -1:
        physical_batch_size = args.batch_size
    else:
        physical_batch_size = args.physical_batch_size

    with BatchMemoryManager(
        data_loader=train_loader, max_physical_batch_size=physical_batch_size, optimizer=optimizer
        ) as new_train_loader:
        for i, (images, target) in enumerate(tqdm(new_train_loader)):

            images = images.to(device).to(memory_format=torch.channels_last)
            target = target.to(device)

            # compute output
            output = model(images)
            
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc1 = accuracy(preds, labels)
            top1_acc.append(acc1)

            # compute gradient and do SGD step
            loss.backward()

            losses.append(loss.item())

            # make sure we take a step after processing the last mini-batch in the
            # epoch to ensure we start the next epoch with a clean state
            if args.disable_dp:
                optimizer.step()
            else:
                optimizer.step(input = [images], criterion = criterion_func_non_reduction, target=target.cuda(non_blocking=True))
            optimizer.zero_grad()

            if i % args.print_freq == 0:
                if not args.disable_dp:
                    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc):.6f} "
                        f"(ε = {epsilon:.2f}, δ = {args.delta})"
                    )
                else:
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc):.6f} "
                    )
    train_duration = datetime.now() - start_time
    return train_duration


def test(args, model, test_loader, device):
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    with open(f"resnet152_{args.dpsgd_mode}.csv", "a+") as f:
        f.write(f"{get_elapsed_time()},{top1_avg}\n")

    gc.collect()
    torch.cuda.empty_cache()
    return np.mean(top1_acc)

def main():  # noqa: C901
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    world_size = 1

    args = parse_args()
    print(str_args(args))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dpsgd_mode == "naive":
        config.dpsgd_mode = config.MODE_NAIVE
    elif args.dpsgd_mode == "reweight":
        config.dpsgd_mode = config.MODE_REWEIGHT
    elif args.dpsgd_mode == "elegant":
        config.dpsgd_mode = config.MODE_ELEGANT

    config.grad_sample_mode = args.grad_sample_mode
    config.adaptive_clipping = True

    config.verbose = args.verbose

    config.model_type = "cnn"
    config.architecture = args.architecture
    config.batch_size = args.batch_size

    B = args.batch_size

    # Prepare model
    if args.architecture == "resnet18":
        model = resnet18()
    elif args.architecture == "resnet50":
        model = resnet50()
    elif args.architecture == "resnet152":
        model = resnet152()
    else:
        model = models.__dict__[args.architecture](
            pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c))
        )

    # Load pretrained model
    # print(model)
    model = load_resnet18_from_objax(model, np.load(args.pretrained_path), True)

    model = model.to(memory_format=torch.channels_last)

    # Prepare dataset
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(
        augmentations + normalize if args.disable_dp else normalize
    )

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        generator=None,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
    )


    model.train()
    model = model.to(args.device)

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

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon = args.epsilon,
            target_delta = args.delta,
            epochs = args.epochs,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
            loss_reduction="mean",
            grad_sample_mode=config.grad_sample_mode
        )

    model.train()
    # print(model)

    torch.cuda.synchronize()
    start = time.time()
    profiler.init_step()

    # Store some logs
    accuracy_per_epoch = []
    time_per_epoch = []

    best_acc1 = 0

    torch.cuda.synchronize()
    start_train = time.time()
    start_timer()

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        train_duration = train(
            args, model, train_loader, optimizer, privacy_engine, epoch, "cuda"
        )
        pause_timer()
        top1_acc = test(args, model, test_loader, "cuda")
        start_timer()

        # remember best acc@1 and save checkpoint
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)

        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(top1_acc))

    torch.cuda.synchronize()
    end_train = time.time()

    time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
    avg_time_per_epoch = sum(time_per_epoch_seconds) / len(time_per_epoch_seconds)
    metrics = {
        "accuracy": best_acc1,
        "accuracy_per_epoch": accuracy_per_epoch,
        "avg_time_per_epoch_str": str(timedelta(seconds=int(avg_time_per_epoch))),
        "time_per_epoch": time_per_epoch_seconds,
    }

    logger.info(
        "\nNote:\n- 'total_time' includes the data loading time, training time and testing time.\n- 'time_per_epoch' measures the training time only.\n"
    )
    logger.info(metrics)
    logger.info(str_args(args))

    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
    print("================= Result ================")
    print(f"(ε = {epsilon:.2f}, δ = {args.delta})")
    print(f"Best acc1 = {best_acc1}")
    print(f"Total time elapsed = {end_train - start_train} sec", flush=True)




def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR10 with DP-SGD")
    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        help="Start epoch",
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
        default=512,
        type=int,
        metavar="N",
        help="mini-batch size for training dataset, this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--physical_batch_size",
        default=-1,
        type=int,
        help="Physical mini-batch size."
        "BatchMemoryManager in Opacus provide a functionality"
        "to split virtual mini-batch into multiple physical mini-batch"
        "It helps to train with large batch size using a limited memory"
        "(Default: -1, which means it same with batch_size)",
    )
    parser.add_argument(
        "--batch_size_test",
        default=2048,
        type=int,
        help="mini-batch size for test dataset",
    )
    parser.add_argument(
        "--print_freq",
        default=10,
        type=int,
        help="Print frequency",
    )
    parser.add_argument(
        "--input_size",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--lr_schedule", default="cos", type=str, help="lr schedules"
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
        "--epsilon",
        type=float,
        default=4.0,
        help="Target epsilon (default: 4.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/cifar10",
        help="Where CIFAR10 is/will be stored",
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
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose.",
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
        "--pretrained_path", type=str, default="", help="Pretrained model path."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
