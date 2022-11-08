#include <vector>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cuda_runtime.h"

bool _batched_gemm_first_run = true;
cudaDeviceProp deviceProp;

////////////////////////////////////////////////////////////////
/*                    Simt kernel                             */

template<typename threadblock_shape, typename mma_shape, typename instruction_shape>
class cutlass_simt_igemm_batched {
public:
  cutlass_simt_igemm_batched() {};
  ~cutlass_simt_igemm_batched() {};

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    threadblock_shape,
    mma_shape,
    instruction_shape,
    cutlass::epilogue::thread::LinearCombination<
      float, 
      1,
      int32_t, 
      float
    >,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2
  >;

  Gemm gemm_op;

  cudaError_t run(
    int m,
    int n,
    int k,
    float alpha,
    int8_t const * const *A,
    int lda,
    int8_t const * const *B,
    int ldb,
    float * const *C,
    int ldc,
    float beta,
    int batch_count,
    cudaStream_t stream) {  

    using namespace cutlass::gemm::device;

    typename Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                        A, cutlass::layout::RowMajor(lda),
                        B, cutlass::layout::ColumnMajor(ldb),
                        C, cutlass::layout::ColumnMajor(ldc),
                        C, cutlass::layout::ColumnMajor(ldc),
                        typename Gemm::EpilogueOutputOp::Params(alpha, beta),
                        batch_count);

    cutlass::Status status = gemm_op(args, nullptr, stream);

    if (status != cutlass::Status::kSuccess) {
      std::cout << cutlassGetStatusString(status) << std::endl;
      return cudaErrorUnknown;
    }

    return cudaSuccess;
  }
};

// A100
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
////////////////////////////////////////////////////////////////
/*                    TensorOp kernel                             */

template<typename threadblock_shape, typename mma_shape, typename instruction_shape>
class cutlass_tensorop_igemm_batched {
public:
  cutlass_tensorop_igemm_batched() {};
  ~cutlass_tensorop_igemm_batched() {};

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    threadblock_shape,
    mma_shape,
    instruction_shape,
    cutlass::epilogue::thread::LinearCombination<
      float, 
      128 / cutlass::sizeof_bits<float>::value,
      int32_t, 
      float
    >
  >;

  Gemm gemm_op;

  cudaError_t run(
    int m,
    int n,
    int k,
    float alpha,
    int8_t const * const *A,
    int lda,
    int8_t const * const *B,
    int ldb,
    float * const *C,
    int ldc,
    float beta,
    int batch_count,
    cudaStream_t stream) {  

    using namespace cutlass::gemm::device;

    typename Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                        A, cutlass::layout::RowMajor(lda),
                        B, cutlass::layout::ColumnMajor(ldb),
                        C, cutlass::layout::ColumnMajor(ldc),
                        C, cutlass::layout::ColumnMajor(ldc),
                        typename Gemm::EpilogueOutputOp::Params(alpha, beta),
                        batch_count);

    cutlass::Status status = gemm_op(args, nullptr, stream);

    if (status != cutlass::Status::kSuccess) {
      std::cout << cutlassGetStatusString(status) << std::endl;
      return cudaErrorUnknown;
    }

    return cudaSuccess;
  }
};
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
// A100
cutlass_tensorop_igemm_batched<
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<32, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>
  > cutlass_tensorop_igemm_batched_128x128x32_32x64x32_16x8x16;

cutlass_tensorop_igemm_batched<
  cutlass::gemm::GemmShape<128, 128, 64>,
  cutlass::gemm::GemmShape<32, 64, 64>,
  cutlass::gemm::GemmShape<8, 8, 16>
  > cutlass_tensorop_igemm_batched_128x128x64_32x64x64_8x8x16;
#endif
// V100
// #else
cutlass_simt_igemm_batched<
  cutlass::gemm::GemmShape<128, 128, 16>,
  cutlass::gemm::GemmShape<64, 64, 16>,
  cutlass::gemm::GemmShape<1, 1, 4>
  > cutlass_simt_igemm_batched_128x128x16_64x64x16_1x1x4;

cutlass_simt_igemm_batched<
  cutlass::gemm::GemmShape<128, 128, 16>,
  cutlass::gemm::GemmShape<128, 32, 16>,
  cutlass::gemm::GemmShape<1, 1, 4>
  > cutlass_simt_igemm_batched_128x128x16_128x32x16_1x1x4;

cutlass_simt_igemm_batched<
  cutlass::gemm::GemmShape<128, 256, 32>,
  cutlass::gemm::GemmShape<32, 128, 32>,
  cutlass::gemm::GemmShape<1, 1, 4>
  > cutlass_simt_igemm_batched_128x256x32_32x128x32_1x1x4;

// #endif

cudaError_t cutlass_simt_igemm_int8_batched_gemm(
  int m,
  int n,
  int k,
  float alpha,
  int8_t const * const *A,
  int lda,
  int8_t const * const *B,
  int ldb,
  float * const *C,
  int ldc,
  float beta,
  int batch_count,
  cudaStream_t stream) {  

  if (_batched_gemm_first_run) {
    cudaGetDeviceProperties(&deviceProp, 0);

    _batched_gemm_first_run = false;
  }

  using cutlass::gemm::GemmShape;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  // A100
  if (k <= 32) {
    return cutlass_tensorop_igemm_batched_128x128x32_32x64x32_16x8x16.run
            (m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            beta,
            batch_count,
            stream);
  }
  else if (k > 32) {
    return cutlass_tensorop_igemm_batched_128x128x64_32x64x64_8x8x16.run
            (m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            beta,
            batch_count,
            stream);
  }
  else {
    return cutlass_tensorop_igemm_batched_128x128x64_32x64x64_8x8x16.run
            (m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            beta,
            batch_count,
            stream);
  }
  return cutlass_tensorop_igemm_batched_128x128x64_32x64x64_8x8x16.run
          (m,
          n,
          k,
          alpha,
          A,
          lda,
          B,
          ldb,
          C,
          ldc,
          beta,
          batch_count,
          stream);

#endif
  // V100
  if (n <= 768 && m <= 768 && k <= 32) {
    return cutlass_simt_igemm_batched_128x128x16_64x64x16_1x1x4.run
            (m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            beta,
            batch_count,
            stream);
  }
  else if ((n > 768 || m > 768) && k > 32) {
    return cutlass_simt_igemm_batched_128x128x16_128x32x16_1x1x4.run
            (m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            beta,
            batch_count,
            stream);
  }
  else {
    return cutlass_simt_igemm_batched_128x256x32_32x128x32_1x1x4.run
            (m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            beta,
            batch_count,
            stream);
  }
// #endif
}