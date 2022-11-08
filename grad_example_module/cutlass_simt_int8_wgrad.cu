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

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//                        Simt Kernel
template<typename threadblock_shape, typename mma_shape>
class Conv2dWgrad{
  using Conv2dWgradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
      int8_t, cutlass::layout::TensorNHWC,
      int8_t, cutlass::layout::TensorNHWC,
      int32_t, cutlass::layout::TensorNHWC,
      int32_t,
      cutlass::arch::OpClassSimt,
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      cutlass::arch::Sm80,
#else
      cutlass::arch::Sm75,
#endif
      threadblock_shape,
      mma_shape,
      cutlass::gemm::GemmShape<1, 1, 4>,
      cutlass::epilogue::thread::LinearCombination<
      float, 1, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized
      >::Kernel;

  using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      float, 1, int32_t, float>;

  /// Reduction kernel
  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
      int32_t,
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

public:
  ImplicitGemm implicit_gemm;
  ReductionDevice reduction_op;

  Conv2dWgrad() {}
  ~Conv2dWgrad() {}

  cudaError_t run(int8_t * ograd,
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
                  cudaStream_t stream) {

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

    cutlass::TensorRef<int8_t, TensorNHWC> tensor_a(ograd, TensorNHWC::packed(cutlass::Tensor4DCoord(N, P, Q, K)));
    cutlass::TensorRef<int8_t, TensorNHWC> tensor_b(actv, TensorNHWC::packed(cutlass::Tensor4DCoord(N, H, W, C)));
    cutlass::TensorRef<float, TensorNHWC> tensor_c(wgrad, TensorNHWC::packed(cutlass::Tensor4DCoord(K, R, S, C)));
    cutlass::TensorRef<float, TensorNHWC> tensor_d(wgrad, TensorNHWC::packed(cutlass::Tensor4DCoord(K, R, S, C)));

    typename ImplicitGemm::Arguments arguments{
      problem_size,
      tensor_a,
      tensor_b,
      {nullptr, TensorNHWC()},
      {nullptr, TensorNHWC()},
      {float(alpha), float(beta)},
      split_k_mode
    };

    // checkCutlassRaw(implicit_gemm.initialize(arguments, workspace, stream));
    checkCutlassRaw(implicit_gemm(arguments, workspace, stream));

    // Do reduction
    static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemm::kConvolutionalOperator;
    typename ReductionDevice::Arguments reduction_args(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
        problem_size.split_k_slices,
        cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
        // Reduction input
        {
            reinterpret_cast<int32_t*> (workspace),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
        },
        // Destination
        {
            tensor_d.data(),
            ReductionStrideIndex(tensor_d.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
        },
        // Source
        {
            tensor_c.data(),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
        },
        {alpha, beta}
    );

    // checkCutlassRaw(reduction_op.initialize(reduction_args, nullptr, stream));
    checkCutlassRaw(reduction_op(reduction_args, nullptr, stream));

    return cudaSuccess;
  }

  size_t get_workspace(
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
    int split_k_slices) {

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

    cutlass::TensorRef<int8_t, TensorNHWC> tensor_a(NULL, TensorNHWC::packed(cutlass::Tensor4DCoord(N, P, Q, K)));
    cutlass::TensorRef<int8_t, TensorNHWC> tensor_b(NULL, TensorNHWC::packed(cutlass::Tensor4DCoord(N, H, W, C)));

    typename ImplicitGemm::Arguments arguments{
      problem_size,
      tensor_a,
      tensor_b,
      {nullptr, TensorNHWC()},
      {nullptr, TensorNHWC()},
      {float(1.0), float(0.0)},
      split_k_mode
    };

    size_t workspace_size = implicit_gemm.get_workspace_size(arguments);

    return workspace_size;
  }
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//                        Tensor Core Kernel
template<typename threadblock_shape, typename mma_shape, typename inst_shape>
class Conv2dWgradTensorOp{
  using Conv2dWgradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
      int8_t, cutlass::layout::TensorNHWC,
      int8_t, cutlass::layout::TensorNHWC,
      int32_t, cutlass::layout::TensorNHWC,
      int32_t,
      cutlass::arch::OpClassTensorOp,
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      cutlass::arch::Sm80,
#else
      cutlass::arch::Sm75,
#endif
      threadblock_shape,
      mma_shape,
      inst_shape,
      cutlass::epilogue::thread::LinearCombination<
      float, 128 / cutlass::sizeof_bits<float>::value, int32_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized
      >::Kernel;

  using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      float, 128 / cutlass::sizeof_bits<float>::value, int32_t, float>;

  /// Reduction kernel
  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
      int32_t,
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

public:
  ImplicitGemm implicit_gemm;
  ReductionDevice reduction_op;

  Conv2dWgradTensorOp() {}
  ~Conv2dWgradTensorOp() {}

  cudaError_t run(int8_t * ograd,
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
                  cudaStream_t stream) {

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

    cutlass::TensorRef<int8_t, TensorNHWC> tensor_a(ograd, TensorNHWC::packed(cutlass::Tensor4DCoord(N, P, Q, K)));
    cutlass::TensorRef<int8_t, TensorNHWC> tensor_b(actv, TensorNHWC::packed(cutlass::Tensor4DCoord(N, H, W, C)));
    cutlass::TensorRef<float, TensorNHWC> tensor_c(wgrad, TensorNHWC::packed(cutlass::Tensor4DCoord(K, R, S, C)));
    cutlass::TensorRef<float, TensorNHWC> tensor_d(wgrad, TensorNHWC::packed(cutlass::Tensor4DCoord(K, R, S, C)));

    typename ImplicitGemm::Arguments arguments{
      problem_size,
      tensor_a,
      tensor_b,
      {nullptr, TensorNHWC()},
      {nullptr, TensorNHWC()},
      {float(alpha), float(beta)},
      split_k_mode
    };

    // checkCutlassRaw(implicit_gemm.initialize(arguments, workspace, stream));
    checkCutlassRaw(implicit_gemm(arguments, workspace, stream));

    // Do reduction
    static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemm::kConvolutionalOperator;
    typename ReductionDevice::Arguments reduction_args(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
        problem_size.split_k_slices,
        cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
        // Reduction input
        {
            reinterpret_cast<int32_t*> (workspace),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
        },
        // Destination
        {
            tensor_d.data(),
            ReductionStrideIndex(tensor_d.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
        },
        // Source
        {
            tensor_c.data(),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::ImplicitGemmKernel::kTensorCStrideIdx])
        },
        {alpha, beta}
    );

    // checkCutlassRaw(reduction_op.initialize(reduction_args, nullptr, stream));
    checkCutlassRaw(reduction_op(reduction_args, nullptr, stream));

    return cudaSuccess;
  }

  size_t get_workspace(
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
    int split_k_slices) {

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

    cutlass::TensorRef<int8_t, TensorNHWC> tensor_a(NULL, TensorNHWC::packed(cutlass::Tensor4DCoord(N, P, Q, K)));
    cutlass::TensorRef<int8_t, TensorNHWC> tensor_b(NULL, TensorNHWC::packed(cutlass::Tensor4DCoord(N, H, W, C)));

    typename ImplicitGemm::Arguments arguments{
      problem_size,
      tensor_a,
      tensor_b,
      {nullptr, TensorNHWC()},
      {nullptr, TensorNHWC()},
      {float(1.0), float(0.0)},
      split_k_mode
    };

    size_t workspace_size = implicit_gemm.get_workspace_size(arguments);

    return workspace_size;
  }
};
#endif

// A100
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
Conv2dWgrad \
<cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 16, 8>> \
cutlass_simt_iwgrad_32x32x16_16x16x8;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 32, 8>, cutlass::gemm::GemmShape<16, 32, 8>> \
cutlass_simt_iwgrad_64x32x8_16x32x8;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 32, 16>, cutlass::gemm::GemmShape<16, 16, 16>> \
cutlass_simt_iwgrad_64x32x16_16x16x16;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 64, 4>, cutlass::gemm::GemmShape<16, 64, 4>> \
cutlass_simt_iwgrad_64x64x4_16x64x4;

// Conv2dWgradTensorOp<
// cutlass::gemm::GemmShape<256, 256, 128>,
// cutlass::gemm::GemmShape<64, 32, 128>,
// cutlass::gemm::GemmShape<8, 8, 16>
// > cutlass_tensorop_iwgrad_256x256x128_64x32x128_8x8x16;

// Conv2dWgradTensorOp<
// cutlass::gemm::GemmShape<256, 256, 128>,
// cutlass::gemm::GemmShape<128, 32, 32>,
// cutlass::gemm::GemmShape<16, 8, 16>
// > cutlass_tensorop_iwgrad_256x256x128_128x32x32_16x8x16;

// Conv2dWgradTensorOp<
// cutlass::gemm::GemmShape<256, 256, 128>,
// cutlass::gemm::GemmShape<128, 64, 64>,
// cutlass::gemm::GemmShape<16, 8, 32>
// > cutlass_tensorop_iwgrad_256x256x128_128x64x64_16x8x32;

// Conv2dWgradTensorOp<
// cutlass::gemm::GemmShape<256, 256, 128>,
// cutlass::gemm::GemmShape<64, 64, 64>,
// cutlass::gemm::GemmShape<16, 8, 16>
// > cutlass_tensorop_iwgrad_256x256x128_64x64x64_16x8x16;

#else
// V100
Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 32, 16>, cutlass::gemm::GemmShape<16, 16, 16>> \
cutlass_simt_iwgrad_64x32x16_16x16x16;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 32, 4>, cutlass::gemm::GemmShape<16, 32, 4>> \
cutlass_simt_iwgrad_64x32x4_16x32x4;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 32, 32>, cutlass::gemm::GemmShape<32, 32, 8>> \
cutlass_simt_iwgrad_64x32x32_32x32x8;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 32, 8>, cutlass::gemm::GemmShape<16, 32, 8>> \
cutlass_simt_iwgrad_64x32x8_16x32x8;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 128, 16>, cutlass::gemm::GemmShape<16, 32, 16>> \
cutlass_simt_iwgrad_64x128x16_16x32x16;

Conv2dWgrad \
<cutlass::gemm::GemmShape<128, 32, 16>, cutlass::gemm::GemmShape<16, 32, 16>> \
cutlass_simt_iwgrad_128x32x16_16x32x16;

Conv2dWgrad \
<cutlass::gemm::GemmShape<64, 128, 8>, cutlass::gemm::GemmShape<32, 32, 8>> \
cutlass_simt_iwgrad_64x128x8_32x32x8;
#endif

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
  float alpha,
  float beta,
  cudaStream_t stream) {  

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if (C <= 3 && H <= 32) {
    return cutlass_simt_iwgrad_32x32x16_16x16x8.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      8, alpha, beta, stream);
  } 
  else if (C <= 3 && H > 32) {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      64, alpha, beta, stream);
  } 
  else if (C <= 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x16_16x16x16.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32, alpha, beta, stream);
  } 
  else if (C <= 64 && K <= 64 && R > 1) {
    return cutlass_simt_iwgrad_64x64x4_16x64x4.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32, alpha, beta, stream);
  } 
  else if (C <= 64 && K > 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32, alpha, beta, stream);
  } 
  else if (C > 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32, alpha, beta, stream);
  } 
  else {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32, alpha, beta, stream);
  }
#else
  if (C <= 3 && H <= 32) {
    return cutlass_simt_iwgrad_64x32x16_16x16x16.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      8, alpha, beta, stream);
  } 
  else if (C <= 3 && H > 32) {
    return cutlass_simt_iwgrad_64x32x4_16x32x4.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      64, alpha, beta, stream);
  } 
  else if (C <= 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x4_16x32x4.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      64, alpha, beta, stream);
  } 
  else if (C <= 64 && K <= 64 && R > 1) {
    return cutlass_simt_iwgrad_64x128x16_16x32x16.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      16, alpha, beta, stream);
  } 
  else if (C <= 64 && K > 64 && R <= 1) {
    return cutlass_simt_iwgrad_128x32x16_16x32x16.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      16, alpha, beta, stream);
  } 
  else if (C > 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x128x8_32x32x8.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32, alpha, beta, stream);
  } 
  else {
    return cutlass_simt_iwgrad_64x128x8_32x32x8.run(
      ograd, actv, wgrad, workspace,
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32, alpha, beta, stream);
  }
#endif
}

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
  int dilation_w) {  

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if (C <= 3 && H <= 32) {
    return cutlass_simt_iwgrad_32x32x16_16x16x8.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      8);
  } 
  else if (C <= 3 && H > 32) {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      64);
  } 
  else if (C <= 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x16_16x16x16.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32);
  } 
  else if (C <= 64 && K <= 64 && R > 1) {
    return cutlass_simt_iwgrad_64x64x4_16x64x4.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32);
  } 
  else if (C <= 64 && K > 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32);
  } 
  else if (C > 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32);
  } 
  else {
    return cutlass_simt_iwgrad_64x32x8_16x32x8.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32);
  }
#else
  if (C <= 3 && H <= 32) {
    return cutlass_simt_iwgrad_64x32x16_16x16x16.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      8);
  } 
  else if (C <= 3 && H > 32) {
    return cutlass_simt_iwgrad_64x32x4_16x32x4.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      64);
  } 
  else if (C <= 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x32x4_16x32x4.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      64);
  } 
  else if (C <= 64 && K <= 64 && R > 1) {
    return cutlass_simt_iwgrad_64x128x16_16x32x16.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      16);
  } 
  else if (C <= 64 && K > 64 && R <= 1) {
    return cutlass_simt_iwgrad_128x32x16_16x32x16.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      16);
  } 
  else if (C > 64 && K <= 64 && R <= 1) {
    return cutlass_simt_iwgrad_64x128x8_32x32x8.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32);
  } 
  else {
    return cutlass_simt_iwgrad_64x128x8_32x32x8.get_workspace(
      N, H, W, C, K, R, S, P, Q,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
      32);
  }
#endif
}