#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cuda_runtime.h"

#include "error_helper.h"

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

  using namespace cutlass::gemm::device;

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
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
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

int main(void) {
  int m = 768;
  int n = 3072;
  int k = 32;
  int batch_count = 32;

  std::vector<void *> host_A_array(batch_count, NULL);
  std::vector<void *> host_B_array(batch_count, NULL);
  std::vector<void *> host_C_array(batch_count, NULL);

  for (int i = 0; i < batch_count; ++i) {
    checkCudaErrors(cudaMalloc(&host_A_array[i], sizeof(int8_t)*m*k));
    checkCudaErrors(cudaMalloc(&host_B_array[i], sizeof(int8_t)*k*n));
    checkCudaErrors(cudaMalloc(&host_C_array[i], sizeof(float)*m*n));
  }

  void * device_A_array;
  void * device_B_array;
  void * device_C_array;
  checkCudaErrors(cudaMalloc(&device_A_array, sizeof(void *)*batch_count));
  checkCudaErrors(cudaMalloc(&device_B_array, sizeof(void *)*batch_count));
  checkCudaErrors(cudaMalloc(&device_C_array, sizeof(void *)*batch_count));
  checkCudaErrors(cudaMemcpy(device_A_array, &host_A_array[0], sizeof(void *)*batch_count, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_B_array, &host_B_array[0], sizeof(void *)*batch_count, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_C_array, &host_C_array[0], sizeof(void *)*batch_count, cudaMemcpyHostToDevice));

  // Warm up
  for (int i = 0; i < 3; ++i) {
    cutlass_simt_igemm_int8_batched_gemm(m, n, k,
                                         1.0,
                                         (int8_t **)device_A_array,
                                         m,
                                         (int8_t **)device_B_array,
                                         k,
                                         (float **)device_C_array,
                                         m,
                                         0.0,
                                         batch_count,
                                         NULL);
  }

  cudaEvent_t events[2];
  checkCudaErrors(cudaEventCreate(&events[0]));
  checkCudaErrors(cudaEventCreate(&events[1]));

  checkCudaErrors(cudaEventRecord(events[0]));

  // Measure runtime_ms
  for (int i = 0; i < 20; ++i) {
    cutlass_simt_igemm_int8_batched_gemm(m, n, k,
                                         1.0,
                                         (int8_t **)device_A_array,
                                         m,
                                         (int8_t **)device_B_array,
                                         k,
                                         (float **)device_C_array,
                                         m,
                                         0.0,
                                         batch_count,
                                         NULL);
  }

  checkCudaErrors(cudaEventRecord(events[1]));
  checkCudaErrors(cudaDeviceSynchronize());

  float runtime_ms = 0.0;
  checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

  std::cout << runtime_ms / 20.0 << std::endl;
} 