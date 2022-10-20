#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>

#include "structure.h"
#include "error_helper.h"
#include "utils.h"

#include <curand_kernel.h>

bool _quantize_first_run = true;
curandState *d_state;

std::mt19937 gen(1234);
std::uniform_int_distribution<unsigned long long> dis(0, ULLONG_MAX);

cublasHandle_t quant_handle;
cudaStream_t quant_stream;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void quantize_int8_cuda_kernel(
    float* m,
    int* max_m_idx,
    int8_t* q,
    int num_elem,
    float * scale,
    unsigned long long seed) {
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < num_elem){
    curandState s;
    curand_init(seed + c, 0, 0, &s);
    if (c == 0) {
      (*scale) = (1 << 6)/(*(m + *max_m_idx));
    }
    auto elem = m[c] * (1 << 6)/(*(m + *max_m_idx));

    q[c] = (int8_t)((elem - (int32_t)elem) < curand_uniform(&s) ? (int32_t)elem : elem + 1);
  }
}

std::vector<std::vector<torch::Tensor>> quantize_int8(std::vector<torch::Tensor>& m_list) {
  if (_quantize_first_run) {
    checkCudaErrors(cudaStreamCreate(&quant_stream));
    checkCUBLAS(cublasCreate(&quant_handle));
    checkCUBLAS(cublasSetStream(quant_handle ,quant_stream));
    checkCUBLAS(cublasSetPointerMode(quant_handle, CUBLAS_POINTER_MODE_DEVICE));

    _quantize_first_run = false;
  }

  c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromExternal(quant_stream, 0));

  std::vector<std::vector<torch::Tensor>> q_list;

  for(auto& m : m_list) {
    auto abs = torch::abs(m);
    auto max_m_idx = torch::empty({1}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt));
    checkCUBLAS(cublasIsamax(quant_handle, abs.numel(), (float *)abs.data_ptr(), 1, (int *)max_m_idx.data_ptr()));
    // auto max_m = torch::abs(m).max();
    // auto scale = (1 << 6)/max_m;
    // auto scale = (1 << 6)/m.flatten().index({max_m_idx});

    // auto m_shape = m.sizes();
    // m = m.reshape({m.sizes()[0], -1});
    auto batch_size = m.sizes()[0];
    auto num_params = m.numel() / batch_size;
    auto num_elem = m.numel();

    const int threads = 1024;
    const dim3 blocks((num_elem + threads - 1) / threads, 1);

    auto q = torch::empty(m.sizes(), torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt8));
    auto scale = torch::empty({1}, torch::TensorOptions().device(torch::kCUDA, 0));

    unsigned long long seed = dis(gen);

    quantize_int8_cuda_kernel<<<blocks, threads, 0, quant_stream>>>(
        (float *) m.data_ptr(),
        (int *)max_m_idx.data_ptr(),
        (int8_t *) q.data_ptr(),
        (int) num_elem,
        (float *)scale.data_ptr(),
        seed);

    q_list.push_back({q, scale});
  }

  c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());

  return q_list;
}

__global__ void int32_to_float32_cuda_kernel(
    int32_t *m,
    float *out,
    int n) {
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n){
    out[c] = (float)m[c];
  }
}

void int32_to_float32(torch::Tensor m, torch::Tensor out, int n) {

  const int threads = 1024;
  const dim3 blocks((n + threads - 1) / threads, 1);

  int32_to_float32_cuda_kernel<<<blocks, threads>>>(
      (int32_t *)m.data_ptr(),
      (float *)out.data_ptr(),
      n);

  return;
}