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
from opt_einsum import contract

from .utils import register_grad_sampler
from opacus import config
from opacus.config import MODE_ELEGANT, MODE_NAIVE, MODE_REWEIGHT
from opacus.custom_tensor import GradOutputs, PerSampleGrads
from opacus.utils.quant_utils import batch_quantization_encode

from opacus.profiler import profiler

@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    profiler.record("Backward weight")

    if config.dpsgd_mode == MODE_ELEGANT:
        if config.quantization:
            m, scale = batch_quantization_encode(backprops, bit=8)
            layer.grad_outputs = [GradOutputs(m)]
            layer.grad_outputs_scale = scale
        else:
            layer.grad_outputs = [GradOutputs(backprops)]

    ret = {}
    if layer.weight.requires_grad_opacus:
        if config.dpsgd_mode == MODE_NAIVE:
            gs = contract("n...i,n...j->nij", backprops, activations)
            ret[layer.weight] = PerSampleGrads(gs)
            profiler.record("Backward weight")
        if config.dpsgd_mode == MODE_REWEIGHT or config.dpsgd_mode == MODE_ELEGANT:
            if (len(activations.shape) == 2):
                layer.weight.grad_sample_norms = [activations.norm(2, dim=1) * backprops.norm(2, dim=1)]
                profiler.record("Clip/reduce")
            elif config.dpsgd_mode == MODE_REWEIGHT:
                gs = contract("n...i,n...j->nij", backprops, activations)
                profiler.record("Backward weight")
                layer.weight.grad_sample_norms = [gs.norm(2, dim=(1, 2))]
                profiler.record("Clip/reduce")
            

    if layer.bias is not None and layer.bias.requires_grad_opacus:
        if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_ELEGANT:
            ret[layer.bias] = PerSampleGrads(contract("n...k->nk", backprops))
            profiler.record("Backward weight")
        if config.dpsgd_mode == MODE_REWEIGHT:
            contracted_backprops = contract("n...k->nk", backprops)
            profiler.record("Backward weight")
            layer.bias.grad_sample_norms = [contracted_backprops.norm(2, dim=1)]
            profiler.record("Clip/reduce")

    return ret
