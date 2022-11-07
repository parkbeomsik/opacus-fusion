#pragma once

#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>

#include <vector>
#include <chrono>
#include <algorithm>

#include "structure.h"
#include "error_helper.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<torch::Tensor>> quantize_int8(std::vector<torch::Tensor>& m_list);

void add_noise(torch::Tensor& tensor, float noise_dev);