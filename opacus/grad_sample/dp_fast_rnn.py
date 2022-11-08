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
from opacus.layers.dp_fast_rnn import DPFASTLSTM

@register_grad_sampler(DPFASTLSTM)
def compute_rnn_grad_sample(
    layer: DPFASTLSTM, activations: list[torch.Tensor], backprops: list[torch.Tensor]
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``DPFASTLSTM`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    profiler.record("Backward weight")

    if config.dpsgd_mode == MODE_ELEGANT:
        if config.quantization:
            if layer.bidirectional:
                b0, b1, a0, a1, a2 = batch_quantization_encode(backprops + activations)
                layer.grad_outputs = [GradOutputs(b0[0]), GradOutputs(b1[0])]
                layer.grad_outputs_scale = [b0[1], b1[1]]
                layer.activations = [a0[0], a1[0], a2[0]]
                layer.activations_scale = [a0[1], a1[1], a1[1]]
            else:
                b0, a0, a1 = batch_quantization_encode(backprops + activations)
                layer.grad_outputs = [GradOutputs(b0[0])]
                layer.grad_outputs_scale = [b0[1]]
                layer.activations = [a0[0], a1[0]]
                layer.activations_scale = [a0[1], a1[1]]
        else:
            if layer.bidirectional:
                layer.grad_outputs = [GradOutputs(backprops[0]), GradOutputs(backprops[1])]
                layer.activations = activations
            else:
                layer.grad_outputs = [GradOutputs(backprops[0])]
                layer.activations = activations

    if layer.bidirectional:
        i_actv, h_actv, h_actv_reverse = activations
        grad_gates, grad_gates_reverse = backprops
    else:
        i_actv, h_actv = activations
        grad_gates, = backprops
    

    ret = {}
    if layer.weight_ih.requires_grad_opacus:
        if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
            gs = PerSampleGrads(contract("n...i,n...j->nij", grad_gates, i_actv, backend="torch"))
            profiler.record("Backward weight")
            if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
                ret[layer.weight_ih] = PerSampleGrads(gs)
            # if config.dpsgd_mode == MODE_REWEIGHT:
            #     layer.weight_ih.grad_sample_norms = [gs.norm(2, dim=(1, 2))]
            #     profiler.record("Clip/reduce")

    if layer.weight_hh.requires_grad_opacus:
        if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
            gs = PerSampleGrads(contract("n...i,n...j->nij", grad_gates, h_actv, backend="torch"))
            profiler.record("Backward weight")
            if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
                ret[layer.weight_hh] = PerSampleGrads(gs)
            # if config.dpsgd_mode == MODE_REWEIGHT:
            #     layer.weight_hh.grad_sample_norms = [gs.norm(2, dim=(1, 2))]
            #     profiler.record("Clip/reduce")            

    if layer.bias is not None and layer.bias.requires_grad_opacus:
        contracted_backprops = PerSampleGrads(contract("n...k->nk", grad_gates, backend="torch"))
        profiler.record("Backward weight")
        # if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_ELEGANT:
        ret[layer.bias] = contracted_backprops
        # if config.dpsgd_mode == MODE_REWEIGHT:
        #     layer.bias.grad_sample_norms = [contracted_backprops.norm(2, dim=1)]
        #     profiler.record("Clip/reduce")

    if layer.bidirectional:
        if layer.weight_ih_reverse.requires_grad_opacus:
            if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
                gs = PerSampleGrads(contract("n...i,n...j->nij", grad_gates, i_actv, backend="torch"))
                profiler.record("Backward weight")
                if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
                    ret[layer.weight_ih_reverse] = PerSampleGrads(gs)
                # if config.dpsgd_mode == MODE_REWEIGHT:
                #     layer.weight_ih_reverse.grad_sample_norms = [gs.norm(2, dim=(1, 2))]
                #     profiler.record("Clip/reduce")

        if layer.weight_hh_reverse.requires_grad_opacus:
            if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
                gs = PerSampleGrads(contract("n...i,n...j->nij", grad_gates, h_actv_reverse, backend="torch"))
                profiler.record("Backward weight")
                if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_REWEIGHT:
                    ret[layer.weight_hh_reverse] = PerSampleGrads(gs)
                # if config.dpsgd_mode == MODE_REWEIGHT:
                #     layer.weight_hh_reverse.grad_sample_norms = [gs.norm(2, dim=(1, 2))]
                #     profiler.record("Clip/reduce")            

        if layer.bias_reverse is not None and layer.bias_reverse.requires_grad_opacus:
            contracted_backprops = PerSampleGrads(contract("n...k->nk", grad_gates_reverse, backend="torch"))
            profiler.record("Backward weight")
            # if config.dpsgd_mode == MODE_NAIVE or config.dpsgd_mode == MODE_ELEGANT:
            ret[layer.bias_reverse] = contracted_backprops
            # if config.dpsgd_mode == MODE_REWEIGHT:
            #     layer.bias_reverse.grad_sample_norms = [contracted_backprops.norm(2, dim=1)]
            #     profiler.record("Clip/reduce")

    return ret
