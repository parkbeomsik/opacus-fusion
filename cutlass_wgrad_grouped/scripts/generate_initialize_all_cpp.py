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
      all_function += f"    initialize_{kernel_name}(operations);\n"

    string = f"""
#include <vector>
#include "initialize_{wgrad_name}_{opclass}_grouped.h"

namespace cutlass_wgrad_grouped {{

void initialize_{wgrad_name}_{opclass}_grouped(std::vector<Operation*>& operations) {{
{all_function}
}}

}} // namepsace cutlass_wgrad_grouped
    """

    return string