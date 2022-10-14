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


from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

from .utils import register_grad_sampler
from opacus import config
from opacus.config import MODE_ELEGANT, MODE_NAIVE, MODE_REWEIGHT
from opacus.custom_tensor import GradOutputs, PerSampleGrads

from opacus.profiler import profiler

@register_grad_sampler(nn.GroupNorm)
def compute_group_norm_grad_sample(
    layer: nn.GroupNorm,
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for GroupNorm

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    backprops = GradOutputs(backprops)
    profiler.record("Backward weight")

    ret = {}
    if layer.weight.requires_grad_opacus:
        gs = PerSampleGrads(torch.einsum("ni...->ni", F.group_norm(activations, layer.num_groups, eps=layer.eps) * backprops))
        if config.dpsgd_mode == MODE_ELEGANT or config.dpsgd_mode == MODE_NAIVE:
            # ret[layer.weight] = PerSampleGrads(contract("ni...->ni", gs, backend="torch"))
            ret[layer.weight] = gs
            profiler.record("Backward weight")
        if config.dpsgd_mode == MODE_REWEIGHT:
            # gs = PerSampleGrads(contract("ni...->ni", gs, backend="torch"))
            # gs = PerSampleGrads(torch.einsum("ni...->ni", gs))
            profiler.record("Backward weight")
            layer.weight.grad_sample_norms = [gs.norm(2, dim=1)]
            profiler.record("Clip/reduce")

    if layer.bias is not None and layer.bias.requires_grad_opacus:
        if config.dpsgd_mode == MODE_ELEGANT or config.dpsgd_mode == MODE_NAIVE:
            # ret[layer.bias] = PerSampleGrads(contract("ni...->ni", backprops, backend="torch"))
            ret[layer.bias] = PerSampleGrads(torch.einsum("ni...->ni", backprops))
            profiler.record("Backward weight")
        if config.dpsgd_mode == MODE_REWEIGHT:
            # backprops = PerSampleGrads(contract("ni...->ni", backprops, backend="torch"))
            backprops = PerSampleGrads(torch.einsum("ni...->ni", backprops))
            profiler.record("Backward weight")
            layer.bias.grad_sample_norms = [backprops.norm(2, dim=1)]
            profiler.record("Clip/reduce")

    # if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
    #     del activations
    #     del backprops
    #     del gs

    return ret
