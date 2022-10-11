def get_string(kernel_names):

    all_function = ""

    for kernel_name in kernel_names:
      all_function += f"    void initialize_{kernel_name}(std::vector<Operation*> &operation_list);\n"

    string = f"""
#include <vector>
#include "base_operation.h"

namespace cutlass_wgrad_grouped {{
{all_function}
    void initialize_swgrad_grouped(std::vector<Operation*>& operations);
}}
    """

    return string