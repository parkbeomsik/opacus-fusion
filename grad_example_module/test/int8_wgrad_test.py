import subprocess
import itertools
import os

from tqdm import tqdm

def get_source_code(problem, tb_shape, mma_shape, inst_shape, split_k_slice):
  return f'''
#include <vector>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

#include "cuda_runtime.h"

#include "error_helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements
// In Wgrad, fp32 accumulation is necessary in practice.
using ElementAccumulator = int32_t;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = int8_t;             // Data type of elements in input tensor
using ElementInputB = int8_t;             // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor
using ElementC = ElementOutput;
using ElementCompute = ElementComputeEpilogue;
using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassSimt;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<{tb_shape[0]}, {tb_shape[1]}, {tb_shape[2]}>; // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<{mma_shape[0]}, {mma_shape[1]}, {mma_shape[2]}>;          // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<{inst_shape[0]}, {inst_shape[1]}, {inst_shape[2]}>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// We need two epilogue functors - one for GEMM and another for the final reduction.
// The epilogue for GEMM is not used, but needed to instantiate the CUTLASS kernel template.
// Note that, when the input is fp16 and accumulation is fp32, the output of GEMM needs to be fp32,
// the final reduction is done in fp32, and the reduction epilogue converts fp32 outputs to fp16.
// Therefore, the output type of the GEMM epilogue is ElementCompute, not ElementOutput.

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOpGEMM = cutlass::epilogue::thread::LinearCombination<
    ElementCompute,                                     // Data type of output matrix.
    1,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// The epilogue functor for reduction. This is the one that is actually used.
using EpilogueOpReduction = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    1,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in lin

using Conv2dWgradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementAccumulator, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOpGEMM,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm
    >::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

using EpilogueOutputOp = EpilogueOpReduction;

/// Reduction kernel
using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
  >;

using ReductionDevice = cutlass::reduction::device::ReduceSplitK<ReductionKernel>;
using ReductionStrideIndex = typename ReductionDevice::StrideIndex;

cudaError_t cutlass_simt_iwgrad(
  int8_t * ograd,
  int8_t * actv,
  float * wgrad,
  void * workspace,
  int N,
  int H,
  int W,
  int C,
  int K,
  int R,
  int S,
  int P,
  int Q,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w,
  int split_k_slices,
  float alpha,
  float beta,
  cudaStream_t stream) {{

  ImplicitGemm implicit_gemm;

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  cutlass::conv::Conv2dProblemSize problem_size(
    N,
    H,
    W,
    C,
    K,
    R,
    S,
    P,
    Q,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    mode,
    split_k_slices
  );

  using cutlass::layout::TensorNHWC;

  cutlass::conv::SplitKMode const split_k_mode = cutlass::conv::SplitKMode::kParallel;

  cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a(ograd, TensorNHWC::packed(cutlass::Tensor4DCoord(N, P, Q, K)));
  cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b(actv, TensorNHWC::packed(cutlass::Tensor4DCoord(N, H, W, C)));
  cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c(wgrad, TensorNHWC::packed(cutlass::Tensor4DCoord(K, R, S, C)));
  cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d(wgrad, TensorNHWC::packed(cutlass::Tensor4DCoord(K, R, S, C)));

  typename ImplicitGemm::Arguments arguments{{
    problem_size,
    tensor_a,
    tensor_b,
    {{nullptr, TensorNHWC()}},
    {{nullptr, TensorNHWC()}},
    {{ElementCompute(alpha), ElementCompute(beta)}},
    split_k_mode
  }};

  checkCutlassRaw(implicit_gemm.initialize(arguments, workspace));
  checkCutlassRaw(implicit_gemm(stream));

  // Do reduction
  ReductionDevice reduction_op;
  static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemm::kConvolutionalOperator;
  typename ReductionDevice::Arguments reduction_args(
      cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
      problem_size.split_k_slices,
      cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
      // Reduction input
      {{
          reinterpret_cast<ElementAccumulator*> (workspace),
          ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
      }},
      // Destination
      {{
          tensor_d.data(),
          ReductionStrideIndex(tensor_d.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
      }},
      // Source
      {{
          tensor_c.data(),
          ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
      }},
      {{alpha, beta}}
  );

  checkCutlassRaw(reduction_op.initialize(reduction_args, nullptr));
  checkCutlassRaw(reduction_op());

  return cudaSuccess;
  }}

size_t cutlass_simt_iwgrad_get_workspace(
  int N,
  int H,
  int W,
  int C,
  int K,
  int R,
  int S,
  int P,
  int Q,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w,
  int split_k_slices) {{  

  ImplicitGemm implicit_gemm;

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  cutlass::conv::Conv2dProblemSize problem_size(
    N,
    H,
    W,
    C,
    K,
    R,
    S,
    P,
    Q,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    mode,
    split_k_slices
  );

  using cutlass::layout::TensorNHWC;

  cutlass::conv::SplitKMode const split_k_mode = cutlass::conv::SplitKMode::kParallel;

  cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a(NULL, TensorNHWC::packed(cutlass::Tensor4DCoord(N, P, Q, K)));
  cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b(NULL, TensorNHWC::packed(cutlass::Tensor4DCoord(N, H, W, C)));

  typename ImplicitGemm::Arguments arguments{{
    problem_size,
    tensor_a,
    tensor_b,
    {{nullptr, TensorNHWC()}},
    {{nullptr, TensorNHWC()}},
    {{ElementCompute(1.0), ElementCompute(0.0)}},
    split_k_mode
  }};

  size_t workspace_size = implicit_gemm.get_workspace_size(arguments);

  return workspace_size;
}}

int main(int argc, char * argv[]) {{
  int N = atoi(argv[1]);
  int H = atoi(argv[2]);
  int W = atoi(argv[3]);
  int C = atoi(argv[4]);
  int K = atoi(argv[5]);
  int R = atoi(argv[6]);
  int S = atoi(argv[7]);
  int P = atoi(argv[8]);
  int Q = atoi(argv[9]);
  int pad_h = atoi(argv[10]);
  int pad_w = atoi(argv[11]);
  int stride_h = atoi(argv[12]);
  int stride_w = atoi(argv[13]);
  int dilation_h = atoi(argv[14]);
  int dilation_w = atoi(argv[15]);
  int split_k_slices = atoi(argv[16]);

  size_t ws_size = cutlass_simt_iwgrad_get_workspace(N,
                                                     H,
                                                     W,
                                                     C,
                                                     K,
                                                     R,
                                                     S,
                                                     P,
                                                     Q,
                                                     pad_h,
                                                     pad_w,
                                                     stride_h,
                                                     stride_w,
                                                     dilation_h,
                                                     dilation_w,
                                                     split_k_slices);

  void * ograd;
  void * actv;
  void * wgrad;
  void * ws;
  checkCudaErrors(cudaMalloc(&ograd, sizeof(int8_t)*N*K*P*Q));
  checkCudaErrors(cudaMalloc(&actv, sizeof(int8_t)*N*C*H*W));
  checkCudaErrors(cudaMalloc(&wgrad, sizeof(float)*K*C*R*S));
  checkCudaErrors(cudaMalloc(&ws, ws_size));

  // Warm up
  for (int i = 0; i < 3; ++i) {{
    cutlass_simt_iwgrad((int8_t *)ograd,
                        (int8_t *)actv,
                        (float *)wgrad,
                        ws,
                        N,
                        H,
                        W,
                        C,
                        K,
                        R,
                        S,
                        P,
                        Q,
                        pad_h,
                        pad_w,
                        stride_h,
                        stride_w,
                        dilation_h,
                        dilation_w,
                        split_k_slices,
                        1.0,
                        0.0,
                        NULL);
  }}

  cudaEvent_t events[2];
  checkCudaErrors(cudaEventCreate(&events[0]));
  checkCudaErrors(cudaEventCreate(&events[1]));

  checkCudaErrors(cudaEventRecord(events[0]));

  // Measure runtime_ms
  for (int i = 0; i < 20; ++i) {{
    cutlass_simt_iwgrad((int8_t *)ograd,
                        (int8_t *)actv,
                        (float *)wgrad,
                        ws,
                        N,
                        H,
                        W,
                        C,
                        K,
                        R,
                        S,
                        P,
                        Q,
                        pad_h,
                        pad_w,
                        stride_h,
                        stride_w,
                        dilation_h,
                        dilation_w,
                        split_k_slices,
                        1.0,
                        0.0,
                        NULL);
  }}

  checkCudaErrors(cudaEventRecord(events[1]));
  checkCudaErrors(cudaDeviceSynchronize());

  float runtime_ms = 0.0;
  checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

  std::cout << runtime_ms / 20.0 << std::endl;
}}

'''

def main():
  tb_shape_0_cand = [16, 32, 64, 128, 256]
  tb_shape_1_cand = [16, 32, 64, 128, 256]
  tb_shape_2_cand = [4, 8, 16]
  mma_shape_0_cand = [16, 32, 64, 128]
  mma_shape_1_cand = [16, 32, 64, 128]
  mma_shape_2_cand = [4, 8, 16]
  inst_shape_0_cand = [1]
  inst_shape_1_cand = [1]
  inst_shape_2_cand = [4]
  split_k_slice_cand = [2, 4, 8, 16, 32, 64]

  tb_shape_0_cand = [32]
  tb_shape_1_cand = [32]
  tb_shape_2_cand = [16]
  mma_shape_0_cand = [16]
  mma_shape_1_cand = [16]
  mma_shape_2_cand = [8]
  inst_shape_0_cand = [1]
  inst_shape_1_cand = [1]
  inst_shape_2_cand = [1]

  # problem = [N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w]
  problems = [

              [1, 32, 32, 3, 64, 7, 7, 16, 16, 3, 3, 2, 2, 1, 1],
              [1, 224, 224, 3, 64, 7, 7, 112, 112, 3, 3, 2, 2, 1, 1],

              [1, 56, 56, 64, 64, 1, 1, 56, 56, 0, 0, 1, 1, 1, 1],
              [1, 56, 56, 64, 64, 3, 3, 56, 56, 1, 1, 1, 1, 1, 1],
              [1, 56, 56, 64, 256, 1, 1, 56, 56, 1, 1, 1, 1, 1, 1],
              [1, 56, 56, 256, 64, 1, 1, 56, 56, 1, 1, 1, 1, 1, 1]
              ]

  for problem in problems:

    all_cases = itertools.product(tb_shape_0_cand, tb_shape_1_cand, tb_shape_2_cand,
                                  mma_shape_0_cand, mma_shape_1_cand, mma_shape_2_cand,
                                  inst_shape_0_cand, inst_shape_1_cand, inst_shape_2_cand,
                                  split_k_slice_cand)
    num_cases = len(tb_shape_0_cand) * len(tb_shape_1_cand) * len(tb_shape_2_cand) *\
                len(mma_shape_0_cand) * len(mma_shape_1_cand) * len(mma_shape_2_cand) *\
                len(inst_shape_0_cand) * len(inst_shape_1_cand) * len(inst_shape_2_cand) *\
                len(split_k_slice_cand)

    min_runtime = 1000.0
    min_conf = None

    for conf in tqdm(all_cases, total=num_cases):
      tb_shape = [conf[0], conf[1], conf[2]]
      mma_shape = [conf[3], conf[4], conf[5]]
      inst_shape = [conf[6], conf[7], conf[8]]
      split_k_slice = conf[9]

      if mma_shape[0] > tb_shape[0] or mma_shape[1] > tb_shape[1] or mma_shape[2] > tb_shape[2]:
        continue

      exec_name = f'int8_wgrad_kernels/simt_iwgrad_{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}_splitK_test'

      if not os.path.isfile(exec_name):       
        # continue

        with open("test_wgrad_main.cu", "w") as f:
          f.write(get_source_code(problem, tb_shape, mma_shape, inst_shape, split_k_slice))

        compile_ret = subprocess.run(f"nvcc -gencode=arch=compute_80,code=compute_80 -I/home/beomsik/dp/cutlass/include test_wgrad_main.cu -o {exec_name}", shell=True, capture_output=True)
        print(compile_ret.stderr)
        if compile_ret.returncode != 0:
          # print(f"{tb_shape}, {mma_shape}, {inst_shape} : compile error")
          continue

      ret = subprocess.run(f"{exec_name} {' '.join(map(lambda x: str(x), problem))} {split_k_slice}", shell=True, capture_output=True)

      # print(f"{tb_shape}, {mma_shape}, {inst_shape} : {ret.stdout.decode()[:-1]} ms")

      # subprocess.run(exec_name, shell=True, capture_output=True)

      try:
        if float(ret.stdout.decode()[:-1]) < min_runtime:
          min_runtime = float(ret.stdout.decode()[:-1])
          min_conf = conf
      except:
        pass

    print(f"{problem} : <{min_conf[0]}, {min_conf[1]}, {min_conf[2]}>, <{min_conf[3]}, {min_conf[4]}, {min_conf[5]}>, <{min_conf[6]}, {min_conf[7]}, {min_conf[8]}>, {min_conf[9]}, {min_runtime} ms")


if __name__ == "__main__":
  main()