#include <vector>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cuda_runtime.h"

template<typename threadblock_shape, typename mma_shape, typename instruction_shape>
cudaError_t cutlass_simt_igemm_int8_batched_gemm_kernel(
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
    >
  >;

  Gemm gemm_op;

  Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                       A, cutlass::layout::RowMajor(lda),
                       B, cutlass::layout::ColumnMajor(ldb),
                       C, cutlass::layout::ColumnMajor(ldc),
                       C, cutlass::layout::ColumnMajor(ldc),
                       Gemm::EpilogueOutputOp::Params(alpha, beta),
                       batch_count);

  cutlass::Status status = gemm_op(args, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    std::cout << cutlassGetStatusString(status) << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

cudaError_t cutlass_simt_igemm_int8_batched_gemm_128x256x32_32x128x32(
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

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombination<
      float, 
      1,
      int32_t, 
      float
    >
  >;

  Gemm gemm_op;

  Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                       A, cutlass::layout::ColumnMajor(lda),
                       B, cutlass::layout::ColumnMajor(ldb),
                       C, cutlass::layout::ColumnMajor(ldc),
                       C, cutlass::layout::ColumnMajor(ldc),
                       Gemm::EpilogueOutputOp::Params(alpha, beta),
                       batch_count);

  cutlass::Status status = gemm_op(args, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    std::cout << cutlassGetStatusString(status) << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

cudaError_t cutlass_simt_igemm_int8_batched_gemm_128x128x16_128x32x16(
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

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<128, 32, 16>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombination<
      float, 
      1,
      int32_t, 
      float
    >
  >;

  Gemm gemm_op;

  Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                       A, cutlass::layout::ColumnMajor(lda),
                       B, cutlass::layout::ColumnMajor(ldb),
                       C, cutlass::layout::ColumnMajor(ldc),
                       C, cutlass::layout::ColumnMajor(ldc),
                       Gemm::EpilogueOutputOp::Params(alpha, beta),
                       batch_count);

  cutlass::Status status = gemm_op(args, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    std::cout << cutlassGetStatusString(status) << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

cudaError_t cutlass_simt_igemm_int8_batched_gemm_128x128x16_64x64x16(
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

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombination<
      float, 
      1,
      int32_t, 
      float
    >
  >;

  Gemm gemm_op;

  Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                       A, cutlass::layout::ColumnMajor(lda),
                       B, cutlass::layout::ColumnMajor(ldb),
                       C, cutlass::layout::ColumnMajor(ldc),
                       C, cutlass::layout::ColumnMajor(ldc),
                       Gemm::EpilogueOutputOp::Params(alpha, beta),
                       batch_count);

  cutlass::Status status = gemm_op(args, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    std::cout << cutlassGetStatusString(status) << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

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

  using cutlass::gemm::GemmShape;

  if (n <= 768 && m <= 768 && k <= 32) {
    return cutlass_simt_igemm_int8_batched_gemm_kernel\
            <GemmShape<128, 128, 16>, GemmShape<64, 64, 16>, GemmShape<1, 1, 4>>\
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
    return cutlass_simt_igemm_int8_batched_gemm_kernel\
            <GemmShape<128, 128, 16>, GemmShape<128, 32, 16>, GemmShape<1, 1, 4>>\
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
    return cutlass_simt_igemm_int8_batched_gemm_kernel\
            <GemmShape<128, 256, 32>, GemmShape<32, 128, 32>, GemmShape<1, 1, 4>>\
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
    return cutlass_simt_igemm_int8_batched_gemm_128x256x32_32x128x32(m,
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
}