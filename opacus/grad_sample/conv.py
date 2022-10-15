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

import math
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import unfold2d, unfold3d
from opacus import config
from opacus.config import MODE_NAIVE, MODE_REWEIGHT, MODE_ELEGANT
from opacus.custom_tensor import GradOutputs, PerSampleGrads
from opacus.utils.quant_utils import batch_quantization_encode
from opt_einsum import contract

from .utils import register_grad_sampler

from opacus.profiler import profiler


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for convolutional layers.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    backprops = GradOutputs(backprops)
    profiler.record("Backward weight")

    origin_mode = config.dpsgd_mode
    if config.model_type == "rnn" and config.dpsgd_mode == MODE_ELEGANT:
        config.dpsgd_mode = MODE_NAIVE

    if config.dpsgd_mode == MODE_ELEGANT:
        if config.quantization:
            m, scale = batch_quantization_encode(backprops, bit=8)
            layer.grad_outputs = [GradOutputs(m)]
            layer.grad_outputs_scale = scale
        else:
            layer.grad_outputs = [GradOutputs(backprops)]

    n = activations.shape[0]
    if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
        # get activations and backprops in shape depending on the Conv layer
        if type(layer) == nn.Conv2d:
            activations = unfold2d(
                activations,
                kernel_size=layer.kernel_size,
                padding=layer.padding,
                stride=layer.stride,
                dilation=layer.dilation,
            )
        elif type(layer) == nn.Conv1d:
            activations = activations.unsqueeze(-2)  # add the H dimension
            # set arguments to tuples with appropriate second element
            if layer.padding == "same":
                total_pad = layer.dilation[0] * (layer.kernel_size[0] - 1)
                left_pad = math.floor(total_pad / 2)
                right_pad = total_pad - left_pad
            elif layer.padding == "valid":
                left_pad, right_pad = 0, 0
            else:
                left_pad, right_pad = layer.padding[0], layer.padding[0]
            activations = F.pad(activations, (left_pad, right_pad))
            activations = torch.nn.functional.unfold(
                activations,
                kernel_size=(1, layer.kernel_size[0]),
                stride=(1, layer.stride[0]),
                dilation=(1, layer.dilation[0]),
            )
        elif type(layer) == nn.Conv3d:
            activations = unfold3d(
                activations,
                kernel_size=layer.kernel_size,
                padding=layer.padding,
                stride=layer.stride,
                dilation=layer.dilation,
            )
    if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
        backprops = backprops.reshape(n, -1, activations.shape[-1])
    if config.dpsgd_mode == MODE_ELEGANT:
        backprops = backprops.view(n, backprops.shape[1], -1)

    ret = {}
    if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
        if layer.weight.requires_grad_opacus:
            # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
            # grad_sample = PerSampleGrads(contract("noq,npq->nop", backprops, activations, backend="torch"))
            grad_sample = PerSampleGrads(torch.einsum("noq,npq->nop", backprops, activations))
            del activations
            # rearrange the above tensor and extract diagonals.
            grad_sample = PerSampleGrads(grad_sample.view(
                n,
                layer.groups,
                -1,
                layer.groups,
                int(layer.in_channels / layer.groups),
                np.prod(layer.kernel_size),
            ))
            # grad_sample = PerSampleGrads(contract("ngrg...->ngr...", grad_sample, backend="torch").contiguous())
            grad_sample = PerSampleGrads(torch.einsum("ngrg...->ngr...", grad_sample).contiguous())
            profiler.record("Backward weight")

            shape = [n] + list(layer.weight.shape)
            if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
                ret[layer.weight] = PerSampleGrads(grad_sample.view(shape))
            # if config.dpsgd_mode == MODE_REWEIGHT:
            #     if type(layer) == nn.Conv2d:
            #         reduce_dims = (1, 2, 3, 4)
            #     elif type(layer) == nn.Conv1d:
            #         reduce_dims = (1, 2, 3) 
            #     layer.weight.grad_sample_norms = [grad_sample.view(shape).norm(2, dim=reduce_dims)]
            #     profiler.record("Clip/reduce")

    if layer.bias is not None and layer.bias.requires_grad_opacus:
        # if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_ELEGANT:
        ret[layer.bias] = PerSampleGrads(torch.sum(backprops, dim=2))
        profiler.record("Backward weight")
        # if config.dpsgd_mode == MODE_REWEIGHT:
        #     backprops = PerSampleGrads(torch.sum(backprops, dim=2))
        #     profiler.record("Backward weight")
        #     layer.bias.grad_sample_norms = [backprops.norm(2, dim=(1,))]
        #     profiler.record("Clip/reduce")

    if config.model_type == "rnn" and config.dpsgd_mode == MODE_NAIVE:
        config.dpsgd_mode = origin_mode

    if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
        del backprops
        del grad_sample

    return ret


# @register_grad_sampler([nn.Conv2d])
def convolution2d_backward_as_a_convolution(
    layer: nn.Conv2d,
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for Conv2d layers using backward.
    This is an alternative implementation and is not used because it is slower in many contexts.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    batch_size = activations.shape[0]
    input_size = activations.shape[1]
    output_size = backprops.shape[1]

    # activations has shape (B, I, H, W)
    # backprops has shape (B, O, H, W)
    activations_ = activations.view(
        batch_size,
        layer.groups,
        input_size // layer.groups,
        activations.shape[2],
        activations.shape[3],
    )  # (B, G, I/G, H, W)

    activations_ = activations_.view(
        activations_.shape[0] * activations_.shape[1],
        activations_.shape[2],
        activations_.shape[3],
        activations_.shape[4],
    )  # (B*G, I / G, H, W)
    activations_ = activations_.transpose(0, 1)  # (I / G, B * G, H, W)
    backprops_ = backprops.view(
        backprops.shape[0] * backprops.shape[1],
        1,
        backprops.shape[2],
        backprops.shape[3],
    )  # (B*O, 1, H, W)

    # Without groups (I, B, H, W) X (B*O, 1, H, W) -> (I, B*O, H, W)
    # With groups (I / G, B*G, H, W) X (B*O, 1, H, W) -> (I / G, B * O, H, W)
    weight_grad_sample = F.conv2d(
        activations_,
        backprops_,
        bias=None,
        dilation=layer.stride,
        padding=layer.padding,
        stride=layer.dilation,
        groups=batch_size * layer.groups,
    )
    weight_grad_sample = weight_grad_sample.view(
        input_size // layer.groups,
        batch_size,
        output_size,
        *weight_grad_sample.shape[-2:]
    )  # (I / G, B, O, H, W)
    weight_grad_sample = weight_grad_sample.movedim(0, 2)  # (B, O, I/G, H, W)
    weight_grad_sample = weight_grad_sample[
        :, :, :, : layer.weight.shape[2], : layer.weight.shape[3]
    ]

    ret = {layer.weight: weight_grad_sample}
    if layer.bias is not None:
        ret[layer.bias] = torch.sum(backprops, dim=[-1, -2])

    return ret
