#pragma once

#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>

#include <vector>
#include <chrono>
#include <algorithm>

#include "structure.h"
#include "error_helper.h"

#include "cutlass_wgrad_grouped.h"

#include "compute_scaling_factor_cuda.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ReturnType get_clip_and_reduced_grads_conv(std::vector<Conv2dConfig> &configs,
                                          std::vector<torch::Tensor>& actvs,
                                          std::vector<torch::Tensor>& ograds,
                                          std::vector<torch::Tensor>& precomputed_per_example_grads,
                                          std::vector<torch::Tensor>& precomputed_per_example_grad_norms,
                                          std::vector<torch::Tensor>& linear_actvs,
                                          std::vector<torch::Tensor>& linear_ograds,
                                        int batch_count = 0,
                                        float max_norm = 1.0,
                                        float noise_multiplier = 1.0,
                                        bool quant = false,
                                        bool verbose = false,
                                        bool profile_time = false,
                                        bool profile_memory = false);