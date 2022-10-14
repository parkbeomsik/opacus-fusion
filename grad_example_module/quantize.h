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

std::vector<torch::Tensor> quantize_int8(torch::Tensor m);

void int32_to_float32(torch::Tensor m, torch::Tensor out, int n);