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

from __future__ import annotations
from collections import defaultdict

import logging
from typing import Callable, List, Optional, Union
import time

import torch
from opacus.optimizers.utils import params
from opacus import config
from opacus.config import MODE_NAIVE, MODE_REWEIGHT, MODE_ELEGANT
from opacus.custom_tensor import PerBatchGrads
from opacus.grad_sample import AbstractGradSampleModule, GradSampleModule
from opacus.utils.module_utils import trainable_modules
from opt_einsum.contract import contract
from torch import nn
from torch.optim import Optimizer

import grad_example_module

from opacus.profiler import profiler, total_ignored_time
from opacus.layers.dp_fast_rnn import DPFASTLSTM


logger = logging.getLogger(__name__)


def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.

    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.

    Notes:
          This is used to only mark ``p.grad_sample`` and ``p.summed_grad``

    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor

    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor or a list of tensors

    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0

    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generated
            (see the notes)

    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).

        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )


class DPOptimizer(Optimizer):
    """
    ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
    sample gradients and add Gaussian noise.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimzer`` assumes that parameters over which it performs optimization belong
    to GradSampleModule and therefore have the ``grad_sample`` attribute.

    On a high level ``DPOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
    2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
    3) Aggregate clipped per sample gradients into ``p.grad``
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    5) Call underlying optimizer to perform optimization step

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = DPOptimizer(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        module: GradSampleModule,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clipping
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError(
                "You must provide expected batch size of the loss reduction is mean"
            )

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator
        self.secure_mode = secure_mode

        self.param_groups = self.original_optimizer.param_groups
        self.defaults = self.original_optimizer.defaults
        self.state = self.original_optimizer.state
        self._step_skip_queue = []
        self._is_last_step_skipped = False

        for p in self.params:
            p.summed_grad = None

        self.first_run = True
        self.module = module

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Return parameter's per sample gradients as a single tensor.

        By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
        batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
        only one batch, and a list of tensors if gradients are accumulated over multiple
        steps. This is done to provide visibility into which sample belongs to which batch,
        and how many batches have been processed.

        This method returns per sample gradients as a single concatenated tensor, regardless
        of how many batches have been accumulated

        Args:
            p: Parameter tensor. Must have ``grad_sample`` attribute

        Returns:
            ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
            concatenating every tensor in ``p.grad_sample`` if it's a list

        Raises:
            ValueError
                If ``p`` is missing ``grad_sample`` attribute
        """

        if not hasattr(p, "grad_sample"):
            raise ValueError(
                "Per sample gradient not found. Are you using GradSampleModule?"
            )
        if p.grad_sample is None:
            raise ValueError(
                "Per sample gradient is not initialized. Not updated in backward pass?"
            )
        if isinstance(p.grad_sample, torch.Tensor):
            ret = p.grad_sample
        elif isinstance(p.grad_sample, list):
            ret = torch.cat(p.grad_sample, dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        return ret

    def signal_skip_step(self, do_skip=True):
        """
        Signals the optimizer to skip an optimization step and only perform clipping and
        per sample gradient accumulation.

        On every call of ``.step()`` optimizer will check the queue of skipped step
        signals. If non-empty and the latest flag is ``True``, optimizer will call
        ``self.clip_and_accumulate``, but won't proceed to adding noise and performing
        the actual optimization step.
        It also affects the behaviour of ``zero_grad()``. If the last step was skipped,
        optimizer will clear per sample gradients accumulated by
        ``self.clip_and_accumulate`` (``p.grad_sample``), but won't touch aggregated
        clipped gradients (``p.summed_grad``)

        Used by :class:`~opacus.utils.batch_memory_manager.BatchMemoryManager` to
        simulate large virtual batches with limited memory footprint.

        Args:
            do_skip: flag if next step should be skipped
        """
        self._step_skip_queue.append(do_skip)

    def _check_skip_next_step(self, pop_next=True):
        """
        Checks if next step should be skipped by the optimizer.
        This is for large Poisson batches that get split into smaller physical batches
        to fit on the device. Batches that do not correspond to the end of a Poisson
        batch or thus `skipped` as their gradient gets accumulated for one big step.
        """
        if self._step_skip_queue:
            if pop_next:
                return self._step_skip_queue.pop(0)
            else:
                return self._step_skip_queue[0]
        else:
            return False

    @property
    def params(self) -> List[nn.Parameter]:
        """
        Returns a flat list of ``nn.Parameter`` managed by the optimizer
        """
        return params(self)

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        """
        Returns a flat list of per sample gradient tensors (one per parameter)
        """
        ret = []
        for p in self.params:
            ret.append(self._get_flat_grad_sample(p))
        return ret

    @property
    def accumulated_iterations(self) -> int:
        """
        Returns number of batches currently accumulated and not yet processed.

        In other words ``accumulated_iterations`` tracks the number of forward/backward
        passed done in between two optimizer steps. The value would typically be 1,
        but there are possible exceptions.

        Used by privacy accountants to calculate real sampling rate.
        """
        if config.dpsgd_mode == MODE_REWEIGHT or config.dpsgd_mode == MODE_ELEGANT:
            return 1
        
        vals = []
        for p in self.params:
            if not hasattr(p, "grad_sample"):
                raise ValueError(
                    "Per sample gradient not found. Are you using GradSampleModule?"
                )
            if isinstance(p.grad_sample, torch.Tensor):
                vals.append(1)
            elif isinstance(p.grad_sample, list):
                vals.append(len(p.grad_sample))
            else:
                raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        if len(set(vals)) > 1:
            raise ValueError(
                "Number of accumulated steps is inconsistent across parameters"
            )
        return vals[0]

    def attach_step_hook(self, fn: Callable[[DPOptimizer], None]):
        """
        Attaches a hook to be executed after gradient clipping/noising, but before the
        actual optimization step.

        Most commonly used for privacy accounting.

        Args:
            fn: hook function. Expected signature: ``foo(optim: DPOptimizer)``
        """

        self.step_hook = fn

    def clip_and_accumulate(self, **kwargs):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if config.dpsgd_mode == MODE_NAIVE:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )

            for p in self.params:
                _check_processed_flag(p.grad_sample)
                grad_sample = self._get_flat_grad_sample(p)
                grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

                if p.summed_grad is not None:
                    p.summed_grad += PerBatchGrads(grad)
                else:
                    p.summed_grad = PerBatchGrads(grad)

                _mark_as_processed(p.grad_sample)

        elif config.dpsgd_mode == MODE_REWEIGHT:
            # Collect gradient norms from all layers
            per_param_norms = []
            for name, layer in trainable_modules(self.module):
                if type(layer) == DPFASTLSTM:
                    if layer.weight_ih.requires_grad_opacus:
                        per_param_norms += layer.weight_ih.grad_sample_norms
                    if layer.weight_hh.requires_grad_opacus:
                        per_param_norms += layer.weight_hh.grad_sample_norms
                    if layer.bidirectional:
                        if layer.weight_ih_reverse.requires_grad_opacus:
                            per_param_norms += layer.weight_ih_reverse.grad_sample_norms
                        if layer.weight_hh_reverse.requires_grad_opacus:
                            per_param_norms += layer.weight_hh_reverse.grad_sample_norms
                else:
                    if layer.weight.requires_grad_opacus:
                        per_param_norms += layer.weight.grad_sample_norms
                if (hasattr(layer, "bias") 
                    and layer.bias is not None 
                    and layer.bias.requires_grad_opacus):
                    per_param_norms += layer.bias.grad_sample_norms
                    if type(layer) == DPFASTLSTM and layer.bidirectional and layer.bias_reverse.requires_grad_opacus:
                        per_param_norms += layer.bias_reverse.grad_sample_norms

            # Compute scaling factors
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )

            input = kwargs["input"]
            criterion = kwargs["criterion"]
            target = kwargs["target"]
            self.module.disable_hooks()

            # PyTorch can't turn-on require_grad for only second backpropagation
            # So, we will do forward propagation again to set require_grad on
            profiler.record("Clip/reduce")

            # This can be removed using better implementation
            # So, we will ignore time for this second propagation
            # torch.cuda.synchronize()
            # start = time.time()

            if config.model_type == "cnn" or config.model_type == "rnn":
                output = self.module(*input)
            if config.model_type == "transformer":
                output = self.module(*input, return_dict=False)[0]

            loss = criterion(output, target)
            loss = (loss * per_sample_clip_factor).mean()

            # profiler.reset_time()
            torch.cuda.synchronize()
            # total_ignored_time.append(time.time() - start)

            ## Second backpropagation (get per-batch gradient)
            loss.backward()

            self.module.enable_hooks()

            for p in self.params:
                p.summed_grad = PerBatchGrads(p.grad)
                p.grad = None

            profiler.record("Clip/reduce")

        elif config.dpsgd_mode == MODE_ELEGANT:
            if config.model_type == "cnn":
                if self.first_run:
                    # Set Conv2d configs
                    self.configs = []
                    i = 0
                    for name, layer in trainable_modules(self.module):
                        if isinstance(layer, nn.Conv2d):
                            self.batch_size = layer.activations[0].shape[0]
                            N = layer.activations[0].shape[0]
                            H = layer.activations[0].shape[2]
                            W = layer.activations[0].shape[3]
                            C = layer.activations[0].shape[1]
                            K = layer.grad_outputs[0].shape[1]
                            R = layer.kernel_size[0]
                            S = layer.kernel_size[1]
                            P = layer.grad_outputs[0].shape[2]
                            Q = layer.grad_outputs[0].shape[3]
                            pad_h, pad_w = layer.padding
                            stride_h, stride_w = layer.stride
                            dilation_h, dilation_w = layer.dilation
                            self.configs.append(grad_example_module.Conv2dConfig(N, H, W, C, K, R, S, P, Q,
                                                                                pad_h, pad_w, stride_h, stride_w,
                                                                                dilation_h, dilation_w, int(P*Q/1024)+1))
                            # args = map(str, [N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w])
                            # print(f"{i} {'x'.join(args)}")
                            # i += 1

                    self.first_run = False

                # Collect activations and grad_outputs, precomputed_grads
                activations = []
                grad_outputs = []
                linear_activations = []
                linear_grad_outputs = []
                linear_norms = []
                precomputed_grads = []
                precomputed_norms = []
                class TTensor(torch.Tensor):
                    pass

                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Conv2d):
                        activations.append(TTensor(layer.activations[0]))
                        grad_outputs.append(layer.grad_outputs[0])
                        if layer.bias is not None:
                            precomputed_grads.append(layer.bias.grad_sample)

                    if isinstance(layer, nn.Linear):
                        # precomputed_grads.append(layer.weight.grad_sample_norm)
                        linear_activations.append(layer.activations[0])
                        linear_grad_outputs.append(layer.grad_outputs[0])
                        linear_norms.append(layer.weight.grad_sample_norms[0])
                        if layer.bias is not None:
                            precomputed_grads.append(layer.bias.grad_sample)
                            precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))

                    if isinstance(layer, nn.GroupNorm):
                        precomputed_grads.append(layer.weight.grad_sample)
                        precomputed_norms.append(layer.weight.grad_sample.norm(2, dim=1))
                        if layer.bias is not None:
                            precomputed_grads.append(layer.bias.grad_sample)
                            precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))

                precomputed_norms = [torch.stack(precomputed_norms + linear_norms, dim=1).norm(2, dim=1)]
                
                profiler.record("Clip/reduce")
                
                # Compute accumulated per-batch gradient and scaling_factors
                result_grad_example_module = \
                    grad_example_module.get_clip_and_reduced_grads_conv(self.configs, activations, grad_outputs,
                                                                precomputed_grads, precomputed_norms,
                                                                linear_activations, linear_grad_outputs,
                                                                self.batch_size, self.max_grad_norm, config.quantization, 
                                                                config.verbose, config.profile_time, config.profile_memory)

                torch.cuda.synchronize()
                profiler.start_interval_time = time.time()
                profiler.add_time_explicit("Backward weight", result_grad_example_module.get_backward_weight_ms())
                profiler.add_time_explicit("Clip/reduce", result_grad_example_module.get_clip_reduce_ms())

                per_batch_grads, per_batch_grads_from_precomputed, per_batch_linear_grads = result_grad_example_module.get_per_batch_grads()

                # Set per-batch grads of key GEMM layers
                idx = 0
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Conv2d):
                        layer.weight.summed_grad = PerBatchGrads(per_batch_grads[idx])
                        idx += 1

                # Set per-batch grads of pre-computed layers
                idx = 0
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Conv2d):
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                    if isinstance(layer, nn.Linear):
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                    if isinstance(layer, nn.GroupNorm):
                        layer.weight.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                        idx += 1
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                # Set per-batch grads of linear layer (weight)
                idx = 0
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Linear):
                        layer.weight.summed_grad = PerBatchGrads(per_batch_linear_grads[idx])
                        idx += 1

                # Clean activations and grad_outputs
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Conv2d):
                        layer.activations = []
                        layer.grad_outputs = []

                    if isinstance(layer, nn.Linear):
                        layer.activations = []
                        layer.grad_outputs = []
                        layer.weight.grad_sample_norms = None

                    if isinstance(layer, nn.GroupNorm):
                        layer.activations = []

            if config.model_type == "transformer":
                if self.first_run:
                    # Set Linear configs
                    self.configs = []
                    self.gemm_key_order = []
                    num_layers_dict = defaultdict(int)
                    for name, layer in trainable_modules(self.module):
                        if isinstance(layer, nn.Linear) and len(layer.activations[0].shape) > 2:
                            self.batch_size = layer.activations[0].shape[0]
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]
                            self.gemm_key_order.append((N, seq_len, in_features, out_features))
                            num_layers_dict[(N, seq_len, in_features, out_features)] += 1

                    self.gemm_key_order = list(set(self.gemm_key_order))
                    self.gemm_key_order = sorted(self.gemm_key_order, key=lambda x: x[2]*x[3])

                    for k in self.gemm_key_order:
                        self.configs.append(grad_example_module.LinearConfig(k[0], k[1], k[2], k[3],
                                                                             num_layers_dict[k]))

                    self.first_run = False
                
                # Collect activations and grad_outputs, precomputed_grads
                activations = [[] for _ in range(len(self.gemm_key_order))]
                grad_outputs = [[] for _ in range(len(self.gemm_key_order))]
                linear_last_activations = []
                linear_last_grad_outputs = []
                linear_last_norms = []
                embedding_activations = []
                embedding_grad_outputs = []
                embedding_vocab_sizes = []
                precomputed_grads = []
                precomputed_norms = []

                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Embedding):
                        embedding_activations.append(layer.activations[0])
                        embedding_grad_outputs.append(layer.grad_outputs[0])
                        embedding_vocab_sizes.append(layer.weight.shape[0])

                    if isinstance(layer, nn.Linear):
                        # Last layer
                        if len(layer.activations[0].shape) == 2:
                            linear_last_activations.append(layer.activations[0])
                            linear_last_grad_outputs.append(layer.grad_outputs[0])
                            linear_last_norms.append(layer.weight.grad_sample_norms[0])
                            if layer.bias is not None:
                                precomputed_grads.append(layer.bias.grad_sample)
                                precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))
                        # Sequential layers
                        else:
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]

                            config_idx = self.gemm_key_order.index((N, seq_len, in_features, out_features))
                            activations[config_idx].append(layer.activations[0])
                            grad_outputs[config_idx].append(layer.grad_outputs[0])
                            if layer.bias is not None:
                                precomputed_grads.append(layer.bias.grad_sample)
                                precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))

                    if isinstance(layer, nn.LayerNorm):
                        precomputed_grads.append(layer.weight.grad_sample)
                        precomputed_norms.append(layer.weight.grad_sample.norm(2, dim=1))
                        if layer.bias is not None:
                            precomputed_grads.append(layer.bias.grad_sample)
                            precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))

                precomputed_norms = [torch.stack(precomputed_norms + linear_last_norms, dim=1).norm(2, dim=1)]
                
                profiler.record("Clip/reduce")

                # Compute accumulated per-batch gradient and scaling_factors
                result_grad_example_module = \
                    grad_example_module.get_clip_and_reduced_grads_linear(self.configs, activations, grad_outputs,
                                                                precomputed_grads, precomputed_norms,
                                                                linear_last_activations, linear_last_grad_outputs,
                                                                embedding_activations, embedding_grad_outputs, embedding_vocab_sizes,
                                                                self.batch_size, self.max_grad_norm, config.quantization, 
                                                                config.verbose, config.profile_time, config.profile_memory)

                torch.cuda.synchronize()
                profiler.start_interval_time = time.time()
                profiler.add_time_explicit("Backward weight", result_grad_example_module.get_backward_weight_ms())
                profiler.add_time_explicit("Clip/reduce", result_grad_example_module.get_clip_reduce_ms())

                per_batch_grads, \
                per_batch_grads_from_precomputed, \
                per_batch_linear_last_grads, \
                per_batch_embedding_grads = result_grad_example_module.get_per_batch_grads()

                # Set per-batch grads of key GEMM layers
                idx = 0
                for gemm_key in self.gemm_key_order:
                    for name, layer in trainable_modules(self.module):
                        if isinstance(layer, nn.Linear) and len(layer.activations[0].shape) > 2:
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]

                            if gemm_key == (N, seq_len, in_features, out_features):
                                layer.weight.summed_grad = PerBatchGrads(per_batch_grads[idx])
                                idx += 1

                # Set per-batch grads of pre-computed layers
                idx = 0
                for name, layer in trainable_modules(self.module):
            
                    if isinstance(layer, nn.Linear):
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                    if isinstance(layer, nn.LayerNorm):
                        layer.weight.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                        idx += 1
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                # Set per-batch grads of last linear layer (weight)
                idx = 0
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Linear):
                        if len(layer.activations[0].shape) == 2:
                            layer.weight.summed_grad = PerBatchGrads(per_batch_linear_last_grads[idx])
                            idx += 1

                # Set per-batch grads of embedding layer
                idx = 0
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Embedding):
                        layer.weight.summed_grad = PerBatchGrads(per_batch_embedding_grads[idx])
                        idx += 1

                # Clean activations and grad_outputs
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Embedding):
                        layer.activations = []
                        layer.grad_outputs = []

                    if isinstance(layer, nn.Linear):
                        layer.activations = []
                        layer.grad_outputs = []
                        layer.weight.grad_sample_norms = None

                    if isinstance(layer, nn.LayerNorm):
                        layer.activations = []

            if config.model_type == "rnn":
                if self.first_run:
                    # Set Linear configs
                    self.configs = []
                    self.gemm_key_order = []
                    num_layers_dict = defaultdict(int)
                    for name, layer in trainable_modules(self.module):
                        if isinstance(layer, nn.Linear) and len(layer.activations[0].shape) > 2:
                            self.batch_size = layer.activations[0].shape[0]
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]
                            self.gemm_key_order.append((N, seq_len, in_features, out_features))
                            num_layers_dict[(N, seq_len, in_features, out_features)] += 1

                        if isinstance(layer, DPFASTLSTM):
                            # weight_ih
                            self.batch_size = layer.activations[0].shape[0]
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]
                            self.gemm_key_order.append((N, seq_len, in_features, out_features))
                            num_layers_dict[(N, seq_len, in_features, out_features)] += 1

                            # weight_hh
                            self.batch_size = layer.activations[1].shape[0]
                            N = layer.activations[1].shape[0]
                            seq_len = layer.activations[1].shape[1]
                            in_features = layer.activations[1].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]
                            self.gemm_key_order.append((N, seq_len, in_features, out_features))
                            num_layers_dict[(N, seq_len, in_features, out_features)] += 1

                            if layer.bidirectional:
                                # weight_ih
                                self.batch_size = layer.activations[0].shape[0]
                                N = layer.activations[0].shape[0]
                                seq_len = layer.activations[0].shape[1]
                                in_features = layer.activations[0].shape[2]
                                out_features = layer.grad_outputs[1].shape[2]
                                self.gemm_key_order.append((N, seq_len, in_features, out_features))
                                num_layers_dict[(N, seq_len, in_features, out_features)] += 1

                                # weight_hh
                                self.batch_size = layer.activations[2].shape[0]
                                N = layer.activations[2].shape[0]
                                seq_len = layer.activations[2].shape[1]
                                in_features = layer.activations[2].shape[2]
                                out_features = layer.grad_outputs[1].shape[2]
                                self.gemm_key_order.append((N, seq_len, in_features, out_features))
                                num_layers_dict[(N, seq_len, in_features, out_features)] += 1

                    self.gemm_key_order = list(set(self.gemm_key_order))
                    self.gemm_key_order = sorted(self.gemm_key_order, key=lambda x: (-100000) * x[1] + x[2]*x[3])

                    print(self.gemm_key_order)

                    for k in self.gemm_key_order:
                        self.configs.append(grad_example_module.LinearConfig(k[0], k[1], k[2], k[3],
                                                                             num_layers_dict[k]))

                    self.first_run = False
                
                # Collect activations and grad_outputs, precomputed_grads
                activations = [[] for _ in range(len(self.gemm_key_order))]
                grad_outputs = [[] for _ in range(len(self.gemm_key_order))]
                linear_last_activations = []
                linear_last_grad_outputs = []
                linear_last_norms = []
                embedding_activations = []
                embedding_grad_outputs = []
                embedding_vocab_sizes = []
                precomputed_grads = []
                precomputed_norms = []

                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Embedding):
                        embedding_activations.append(layer.activations[0])
                        embedding_grad_outputs.append(layer.grad_outputs[0])
                        embedding_vocab_sizes.append(layer.weight.shape[0])

                    if isinstance(layer, nn.Linear):
                        # Last layer
                        if len(layer.activations[0].shape) == 2:
                            linear_last_activations.append(layer.activations[0])
                            linear_last_grad_outputs.append(layer.grad_outputs[0])
                            linear_last_norms.append(layer.weight.grad_sample_norms[0])
                            if layer.bias is not None:
                                precomputed_grads.append(layer.bias.grad_sample)
                                precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))
                        # Sequential layers
                        else:
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]

                            config_idx = self.gemm_key_order.index((N, seq_len, in_features, out_features))
                            activations[config_idx].append(layer.activations[0])
                            grad_outputs[config_idx].append(layer.grad_outputs[0])
                            if layer.bias is not None:
                                precomputed_grads.append(layer.bias.grad_sample)
                                precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))

                    if isinstance(layer, DPFASTLSTM):
                        # weight_ih
                        N = layer.activations[0].shape[0]
                        seq_len = layer.activations[0].shape[1]
                        in_features = layer.activations[0].shape[2]
                        out_features = layer.grad_outputs[0].shape[2]

                        config_idx = self.gemm_key_order.index((N, seq_len, in_features, out_features))
                        activations[config_idx].append(layer.activations[0])
                        grad_outputs[config_idx].append(layer.grad_outputs[0])

                        # weight_hh
                        N = layer.activations[1].shape[0]
                        seq_len = layer.activations[1].shape[1]
                        in_features = layer.activations[1].shape[2]
                        out_features = layer.grad_outputs[0].shape[2]

                        config_idx = self.gemm_key_order.index((N, seq_len, in_features, out_features))
                        activations[config_idx].append(layer.activations[1])
                        grad_outputs[config_idx].append(layer.grad_outputs[0])

                        # bias
                        if layer.bias is not None:
                            precomputed_grads.append(layer.bias.grad_sample)
                            precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))

                        # reverse
                        if layer.bidirectional:
                            # weight_ih_reverse
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[1].shape[2]

                            config_idx = self.gemm_key_order.index((N, seq_len, in_features, out_features))
                            activations[config_idx].append(layer.activations[0])
                            grad_outputs[config_idx].append(layer.grad_outputs[1])

                            # weight_hh_reverse
                            N = layer.activations[2].shape[0]
                            seq_len = layer.activations[2].shape[1]
                            in_features = layer.activations[2].shape[2]
                            out_features = layer.grad_outputs[1].shape[2]

                            config_idx = self.gemm_key_order.index((N, seq_len, in_features, out_features))
                            activations[config_idx].append(layer.activations[2])
                            grad_outputs[config_idx].append(layer.grad_outputs[1])

                            # bias_reverse
                            if layer.bias_reverse is not None:
                                precomputed_grads.append(layer.bias_reverse.grad_sample)
                                precomputed_norms.append(layer.bias_reverse.grad_sample.norm(2, dim=1))

                    if isinstance(layer, nn.GroupNorm) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
                        norm_dim = list(range(1, len(layer.weight.grad_sample.shape)))
                        precomputed_grads.append(layer.weight.grad_sample)
                        precomputed_norms.append(layer.weight.grad_sample.norm(2, dim=norm_dim))
                        if layer.bias is not None:
                            precomputed_grads.append(layer.bias.grad_sample)
                            precomputed_norms.append(layer.bias.grad_sample.norm(2, dim=1))

                precomputed_norms = [torch.stack(precomputed_norms + linear_last_norms, dim=1).norm(2, dim=1)]
                
                profiler.record("Clip/reduce")

                # Compute accumulated per-batch gradient and scaling_factors
                result_grad_example_module = \
                    grad_example_module.get_clip_and_reduced_grads_linear(self.configs, activations, grad_outputs,
                                                                precomputed_grads, precomputed_norms,
                                                                linear_last_activations, linear_last_grad_outputs,
                                                                embedding_activations, embedding_grad_outputs, embedding_vocab_sizes,
                                                                self.batch_size, self.max_grad_norm, config.quantization, 
                                                                config.verbose, config.profile_time, config.profile_memory)

                torch.cuda.synchronize()
                profiler.start_interval_time = time.time()
                profiler.add_time_explicit("Backward weight", result_grad_example_module.get_backward_weight_ms())
                profiler.add_time_explicit("Clip/reduce", result_grad_example_module.get_clip_reduce_ms())

                per_batch_grads, \
                per_batch_grads_from_precomputed, \
                per_batch_linear_last_grads, \
                per_batch_embedding_grads = result_grad_example_module.get_per_batch_grads()

                # Set per-batch grads of key GEMM layers
                idx = 0
                for gemm_key in self.gemm_key_order:
                    for name, layer in trainable_modules(self.module):
                        if isinstance(layer, nn.Linear) and len(layer.activations[0].shape) > 2:
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]

                            if gemm_key == (N, seq_len, in_features, out_features):
                                layer.weight.summed_grad = PerBatchGrads(per_batch_grads[idx])
                                idx += 1

                        if isinstance(layer, DPFASTLSTM):
                            # weight_ih
                            N = layer.activations[0].shape[0]
                            seq_len = layer.activations[0].shape[1]
                            in_features = layer.activations[0].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]

                            if gemm_key == (N, seq_len, in_features, out_features):
                                layer.weight_ih.summed_grad = PerBatchGrads(per_batch_grads[idx])
                                idx += 1

                            # weight_hh
                            N = layer.activations[1].shape[0]
                            seq_len = layer.activations[1].shape[1]
                            in_features = layer.activations[1].shape[2]
                            out_features = layer.grad_outputs[0].shape[2]

                            if gemm_key == (N, seq_len, in_features, out_features):
                                layer.weight_hh.summed_grad = PerBatchGrads(per_batch_grads[idx])
                                idx += 1

                            # reverse
                            if layer.bidirectional:
                                # weight_ih_reverse
                                N = layer.activations[0].shape[0]
                                seq_len = layer.activations[0].shape[1]
                                in_features = layer.activations[0].shape[2]
                                out_features = layer.grad_outputs[1].shape[2]

                                if gemm_key == (N, seq_len, in_features, out_features):
                                    layer.weight_ih_reverse.summed_grad = PerBatchGrads(per_batch_grads[idx])
                                    idx += 1

                                # weight_hh_reverse
                                N = layer.activations[2].shape[0]
                                seq_len = layer.activations[2].shape[1]
                                in_features = layer.activations[2].shape[2]
                                out_features = layer.grad_outputs[1].shape[2]

                                if gemm_key == (N, seq_len, in_features, out_features):
                                    layer.weight_hh_reverse.summed_grad = PerBatchGrads(per_batch_grads[idx])
                                    idx += 1

                # Set per-batch grads of pre-computed layers
                idx = 0
                for name, layer in trainable_modules(self.module):
            
                    if isinstance(layer, nn.Linear):
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                    if isinstance(layer, DPFASTLSTM):
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                        if layer.bias_reverse is not None:
                            layer.bias_reverse.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                    if isinstance(layer, nn.GroupNorm) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
                        layer.weight.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                        idx += 1
                        if layer.bias is not None:
                            layer.bias.summed_grad = PerBatchGrads(per_batch_grads_from_precomputed[idx])
                            idx += 1

                # Set per-batch grads of last linear layer (weight)
                idx = 0
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Linear):
                        if len(layer.activations[0].shape) == 2:
                            layer.weight.summed_grad = PerBatchGrads(per_batch_linear_last_grads[idx])
                            idx += 1

                # Set per-batch grads of embedding layer
                idx = 0
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Embedding):
                        layer.weight.summed_grad = PerBatchGrads(per_batch_embedding_grads[idx])
                        idx += 1

                # Clean activations and grad_outputs
                for name, layer in trainable_modules(self.module):

                    if isinstance(layer, nn.Embedding):
                        layer.activations = []
                        layer.grad_outputs = []

                    if isinstance(layer, nn.Linear):
                        layer.activations = []
                        layer.grad_outputs = []
                        layer.weight.grad_sample_norms = None

                    if isinstance(layer, DPFASTLSTM):
                        layer.activations = []
                        layer.grad_outputs = []

                    if isinstance(layer, nn.GroupNorm) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv2d):
                        layer.activations = []

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad``, ``p.grad_sample`` and ``p.summed_grad`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """

        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        for p in self.params:
            p.grad_sample = None

            if not self._is_last_step_skipped:
                p.summed_grad = None

        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None,
        **kwargs
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate(**kwargs)
        profiler.record("Clip/reduce")

        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        profiler.record("Add noise")

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(**kwargs):
            return self.original_optimizer.step()
        else:
            return None

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.original_optimizer.load_state_dict(state_dict)
