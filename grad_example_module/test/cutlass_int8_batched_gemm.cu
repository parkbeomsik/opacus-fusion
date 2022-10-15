#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cuda_runtime.h"

cudaError_t cutlass_array_igemm_int8(
  int m,
  int n,
  int k,
  float alpha,
  int8_t const * const *A,
  int lda,
  int8_t const * const *B,
  int ldb,
  int32_t * const *C,
  int ldc,
  float beta,
  int batch_count) {  

  using namespace cutlass::gemm::device;

//   using Gemm = cutlass::gemm::device::GemmArray<
//     float, cutlass::layout::ColumnMajor,
//     float, cutlass::layout::ColumnMajor,
//     float, cutlass::layout::ColumnMajor
//   >;

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassWmmaTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<8, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
      int32_t, 
      128 / cutlass::sizeof_bits<int32_t>::value,
      int32_t, 
      int32_t
    >
  >;

  Gemm gemm_op;

  cutlass::Status status = cutlass::Status::kSuccess;
//    = gemm_op({
//     {m, n, k},
//     A, lda,
//     B, ldb,
//     C, ldc,
//     C, ldc,
//     {alpha, beta},
//     batch_count
//   });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

int main(void) {
    cutlass_array_igemm_int8(
    1024,
    1024,
    32,
    1.0,
    NULL,
    1024,
    NULL,
    1024,
    NULL,
    1024,
    0.0,
    100);
}