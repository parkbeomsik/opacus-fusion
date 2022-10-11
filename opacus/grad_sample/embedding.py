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

from .utils import register_grad_sampler

from opacus import config
from opacus.config import MODE_ELEGANT, MODE_NAIVE, MODE_REWEIGHT
from opacus.custom_tensor import GradOutputs, PerSampleGrads
from opacus.profiler import profiler
from opacus.utils.quant_utils import batch_quantization_encode

@register_grad_sampler(nn.Embedding)
def compute_embedding_grad_sample(
    layer: nn.Embedding, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Embedding`` layer.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """    
    if config.dpsgd_mode == MODE_ELEGANT:
        if config.quantization:
            m, scale = batch_quantization_encode(backprops, bit=8)
            layer.grad_outputs = [GradOutputs(m)]
            layer.grad_outputs_scale = scale
        else:
            layer.grad_outputs = [GradOutputs(backprops)]

    ret = {}
    if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
        if layer.weight.requires_grad_opacus:
            saved = torch.backends.cudnn.deterministic
            torch.backends.cudnn.deterministic = True

            batch_size = activations.shape[0]
            index = (
                activations.unsqueeze(-1)
                .expand(*activations.shape, layer.embedding_dim)
                .reshape(batch_size, -1, layer.embedding_dim)
            )
            grad_sample = torch.zeros(
                batch_size, *layer.weight.shape, device=layer.weight.device
            )
            grad_sample.scatter_add_(
                1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
            )
            torch.backends.cudnn.deterministic = saved
            if config.dpsgd_mode == MODE_NAIVE:
                ret[layer.weight] = grad_sample

            profiler.record("Backward weight")

            if config.dpsgd_mode == MODE_REWEIGHT:
                layer.weight.grad_sample_norms = [grad_sample.norm(2, dim=(1, 2))]
                profiler.record("Clip/reduce")

    return ret
