#pragma once

#include <vector>
#include "initialize_swgrad_grouped.h"

namespace cutlass_wgrad_grouped {
    void initialize_swgrad_grouped(std::vector<Operation*>& operations);
}