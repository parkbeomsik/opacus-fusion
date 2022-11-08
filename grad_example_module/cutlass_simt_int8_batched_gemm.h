#include <vector>

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
  cudaStream_t stream = nullptr);