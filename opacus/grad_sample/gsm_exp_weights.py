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

import torch
import torch.nn as nn
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN, RNNLinear
from functools import partial
from typing import List, Tuple, Union

import opacus.config as config
from opacus.config import MODE_NAIVE, MODE_REWEIGHT, MODE_ELEGANT
from opacus.profiler import profiler
from opacus.utils.module_utils import (
    requires_grad,
    trainable_modules,
    trainable_parameters,
)

from opacus.profiler import profiler
from opacus.layers import dp_fast_rnn
from opacus.layers.dp_fast_rnn import DPFASTLSTM

API_CUTOFF_VERSION = "1.13.0.dev"


class GradSampleModuleExpandedWeights(AbstractGradSampleModule):
    """
    ExpandedWeights-based implementation of AbstractGradSampleModule

    Computes per-sample gradients using PyTorch built-in mechanism of ExpandedWeights.
    See README.md for more details
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
    ):
        if not batch_first:
            raise NotImplementedError

        if torch.__version__ >= API_CUTOFF_VERSION:
            from torch.nn.utils._per_sample_grad import call_for_per_sample_grads

            self.call_for_per_sample_grads = call_for_per_sample_grads
        else:
            raise ImportError(
                f"Requested grad_sample_mode=ew, "
                f"but found PyTorch version={torch.__version__}. "
                f"ExpandedWeights available for torch>={API_CUTOFF_VERSION} "
                f"Please install recent PyTorch or use grad_sample_mode=hooks"
            )

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )

        for _, p in m.named_parameters():
            if p.requires_grad:
                p.requires_grad_opacus = True
            else:
                p.requires_grad_opacus = False

        if (config.dpsgd_mode == MODE_REWEIGHT
            or config.dpsgd_mode == MODE_ELEGANT):
            self.add_hooks(
                loss_reduction=loss_reduction,
                batch_first=batch_first,
            )

    def add_hooks(
        self,
        *,
        loss_reduction: str = "mean",
        batch_first: bool = True,
    ) -> None:
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        for _module_name, module in trainable_modules(self._module):
            # Do not add hooks to DPRNN, DPLSTM or DPGRU as the hooks are handled by the `RNNLinear`
            if type(module) in [DPRNN, DPLSTM, DPGRU]:
                continue

            self.autograd_grad_sample_hooks.append(
                module.register_forward_hook(self.capture_activations_hook)
            )

            self.autograd_grad_sample_hooks.append(
                module.register_backward_hook(
                    partial(
                        self.capture_backprops_hook,
                        loss_reduction=loss_reduction,
                        batch_first=batch_first,
                    )
                )
            )

        self.enable_hooks()    


    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.call_for_per_sample_grads(
            module=self._module,
            batch_size=x.shape[0],
            loss_reduction=self.loss_reduction,
        )(x, *args, **kwargs)


    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
        ):
            return

        if not self.hooks_enabled:
            return

        if not hasattr(module, "activations"):
            module.activations = []

        if isinstance(module, DPFASTLSTM):
            module.activations += dp_fast_rnn.input_actvs
            dp_fast_rnn.input_actvs = []
        else:
            module.activations.append(forward_input[0].detach())  # pyre-ignore

        for _, p in trainable_parameters(module):
            p._forward_counter += 1


    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        # For reweight DP-SGD, compute norm of each parameters
        if config.dpsgd_mode == MODE_REWEIGHT:
            profiler.record("Backward weight")
            for _, p in trainable_parameters(module):
                if p._forward_counter == 0 and p.requires_grad_opacus:
                    p.grad_sample_norms = [p.grad_sample.norm(2, dim=list(range(1, len(p.grad_sample.shape))))]

                    del p.grad_sample

            profiler.record("Clip and reduce")


    def disable_hooks(self) -> None:
        if self.hooks_enabled:
            # Create requires_grad to turn-on per-batch gradient computation
            for _, p in trainable_parameters(self._module):
                if p.requires_grad_opacus:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.hooks_enabled = False


    def enable_hooks(self) -> None:
        if not self.hooks_enabled:
            # Create requires_grad_opacus to turn-off per-batch gradient computation
            for _, p in self._module.named_parameters():
                if p.requires_grad:
                    p.requires_grad_opacus = True
                else:
                    p.requires_grad_opacus = False

                p.requires_grad = False
        
        self.hooks_enabled = True