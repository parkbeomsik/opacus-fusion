
#pragma once

#include <vector>
#include "cutlass_wgrad_grouped.h"

namespace cutlass_wgrad_grouped {
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x64x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x64_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x64_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x64_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x128x128_64x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x64_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x64_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x64_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_64x256x128_64x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x64_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x64_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x64_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_128x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_128x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x64x128_128x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x64_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_64x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x128x128_128x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x64_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_64x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_128x256x128_128x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x64_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x64_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x64_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_128x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_128x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x64x128_128x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x64_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_64x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x128x128_128x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x64_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_64x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x64x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x64x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x64x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x64x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x64x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x64x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x128x64_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x128x64_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x128x64_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x128x128_8x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x128x128_16x8x16_3_nhwc(std::vector<Operation*> &operation_list);
    void initialize_cutlass_tensorop_iwgrad_grouped_optimized_256x256x128_128x128x128_16x8x32_3_nhwc(std::vector<Operation*> &operation_list);

    void initialize_iwgrad_tensorop_grouped(std::vector<Operation*>& operations);
}
    