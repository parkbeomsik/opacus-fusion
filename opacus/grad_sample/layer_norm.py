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
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n
from opt_einsum import contract

from .utils import register_grad_sampler

from opacus import config
from opacus.config import MODE_ELEGANT, MODE_NAIVE, MODE_REWEIGHT
from opacus.custom_tensor import GradOutputs, PerSampleGrads
from opacus.profiler import profiler

@register_grad_sampler(nn.LayerNorm)
def compute_layer_norm_grad_sample(
    layer: nn.LayerNorm,
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for LayerNorm

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    backprops = GradOutputs(backprops)
    profiler.record("Backward weight")

    ret = {}
    if layer.weight.requires_grad_opacus:
        gs = PerSampleGrads(sum_over_all_but_batch_and_last_n(
            F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
            * backprops,
            layer.weight.dim(),
        ))
        profiler.record("Backward weight")
        if config.dpsgd_mode == MODE_ELEGANT or config.dpsgd_mode == MODE_NAIVE:
            ret[layer.weight] = PerSampleGrads(gs)
            
        if config.dpsgd_mode == MODE_REWEIGHT:
            layer.weight.grad_sample_norms = [gs.norm(2, dim=1)]
            profiler.record("Clip/reduce")

    if layer.bias.requires_grad_opacus:
        if config.dpsgd_mode == MODE_ELEGANT or config.dpsgd_mode == MODE_NAIVE:
            ret[layer.bias] = PerSampleGrads(sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim()))
            profiler.record("Backward weight")
        if config.dpsgd_mode == MODE_REWEIGHT:
            backprops = PerSampleGrads(sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim()))
            profiler.record("Backward weight")
            layer.bias.grad_sample_norms = [backprops.norm(2, dim=1)]
            profiler.record("Clip/reduce")

    return ret
