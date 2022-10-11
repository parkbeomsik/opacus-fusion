def get_string(kernel_names):

    all_function = ""

    for kernel_name in kernel_names:
      all_function += f"    initialize_{kernel_name}(operations);\n"

    string = f"""
#include <vector>
#include "base_operation.h"
#include "initialize_all.h"

namespace cutlass_wgrad_grouped {{

void initialize_swgrad_grouped(std::vector<Operation*>& operations) {{
{all_function}
}}

}} // namepsace cutlass_wgrad_grouped
    """

    return string