def get_kernel_name(dtype, op_class, tb_shape, mma_shape, inst_shape, num_stages):
    kernel_name = f"cutlass_"
    kernel_name += f"{'simt' if op_class == 'OpClassSimt' else 'tensorop'}_"
    kernel_name += "s" if dtype == "float" else "i"
    kernel_name += "wgrad_grouped_optimized_"
    kernel_name += f"{str(tb_shape[0])}x{str(tb_shape[1])}x{str(tb_shape[2])}_"
    kernel_name += f"{str(mma_shape[0])}x{str(mma_shape[1])}x{str(mma_shape[2])}_"
    kernel_name += f"{str(inst_shape[0])}x{str(inst_shape[1])}x{str(inst_shape[2])}_"
    kernel_name += f"{num_stages}_"
    kernel_name += "nhwc"

    return kernel_name

def get_string(dtype, op_class, tb_shape, mma_shape, inst_shape, num_stages):

    kernel_name = get_kernel_name(dtype, op_class, tb_shape, mma_shape, inst_shape, num_stages)
    
    if dtype == "float":
        element_input = "float"
        element_output = "float"
    elif dtype == "int":
        element_input = "int8_t"
        element_outtput = "int32_t"

    string = f"""
/*
    Generated by generate_cutlass_code.py - Do not edit.
*/

#include "wgrad_grouped_operation.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution_grouped.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad_grouped.h"
#include "cutlass/conv/device/implicit_gemm_convolution_grouped.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

////////////////////////////////////////////////////////////////////

using {kernel_name}_base = typename cutlass::conv::kernel::DefaultConv2dWgradGrouped<
    {element_input}, 
    cutlass::layout::TensorNHWC,
    {element_input},
    cutlass::layout::TensorNHWC,
    {element_output}, cutlass::layout::TensorNHWC,
    {element_output}, 
    cutlass::arch::{op_class}, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<{tb_shape[0]}, {tb_shape[1]}, {tb_shape[2]}>,
    cutlass::gemm::GemmShape<{mma_shape[0]}, {mma_shape[1]}, {mma_shape[2]}>,
    cutlass::gemm::GemmShape<{inst_shape[0]}, {inst_shape[1]}, {inst_shape[2]}>,
    cutlass::epilogue::thread::LinearCombination<
        {element_output}, 1,
        {element_output}, {element_output}>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    {num_stages},
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
    >::Conv2dWgradKernel;

// Derived class
struct {kernel_name} : 
  public {kernel_name}_base {{ }};


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_wgrad_grouped {{

// Initialize all instances
void initialize_{kernel_name}(std::vector<Operation *> &operation_list) {{


  using Operation_{kernel_name} = cutlass::conv::device::ImplicitGemmConvolutionGrouped<
    {kernel_name}>;

  operation_list.push_back(new Conv2dOperation<
    Operation_{kernel_name}>(
      "{kernel_name}"));


}}


///////////////////////////////////////////////////////////////////////////////////////////////////

}} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
    """

    return string