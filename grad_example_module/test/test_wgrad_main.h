#pragma once

#include "templates/conv2d_wgrad_tensorop.h"
#include "templates/operation.h"

void initialize_cutlass_tensorop_iwgrad_grouped_64x64x4_16x64x4(
  std::vector<Operation *>& ops);

void initialize_conv2d_wgrad_tensorop (std::vector<Operation *>& ops) {
    initialize_cutlass_tensorop_iwgrad_grouped_64x64x4_16x64x4(ops);
}