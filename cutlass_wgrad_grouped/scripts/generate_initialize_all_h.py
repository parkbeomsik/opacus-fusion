def get_string(kernel_names, dtype, opclass):

    if opclass == "simt":
        opclass_type = "OpClassSimt"
    elif opclass == "tensorop":
        opclass_type = "OpClassTensorOp"

    if dtype == "int":
        wgrad_name = "iwgrad"
    elif dtype == "float":
        wgrad_name = "swgrad"

    all_function = ""

    for kernel_name in kernel_names:
      all_function += f"    void initialize_{kernel_name}(std::vector<Operation*> &operation_list);\n"

    string = f"""
#pragma once

#include <vector>
#include "cutlass_wgrad_grouped.h"

namespace cutlass_wgrad_grouped {{
{all_function}
    void initialize_{wgrad_name}_{opclass}_grouped(std::vector<Operation*>& operations);
}}
    """

    return string