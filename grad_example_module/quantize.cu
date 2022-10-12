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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void quantize_int8_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> m,
    const float * scale,
    torch::PackedTensorAccessor32<int8_t,2,torch::RestrictPtrTraits> q,
    unsigned long long seed) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < m.size(1)){
    curandState s;
    curand_init(seed + n * m.size(1) * c, 0, 0, &s);
    auto elem = m[n][c] * (*scale);

    q[n][c] = (int8_t)((elem - (int32_t)elem) < curand_uniform(&s) ? (int32_t)elem : elem + 1);
  }
}

std::vector<torch::Tensor> quantize_int8(torch::Tensor m) {
  if (_quantize_first_run) {
    _quantize_first_run = false;
  }


  auto max_m = torch::abs(m).max();
  auto scale = pow(2.0, 7 - 1)/max_m;

  auto m_shape = m.sizes();
  m = m.reshape({m.sizes()[0], -1});
  auto batch_size = m.sizes()[0];
  auto num_params = m.sizes()[1];

  const int threads = 1024;
  const dim3 blocks((num_params + threads - 1) / threads, batch_size);

  auto q = torch::empty({batch_size, num_params}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt8));

  unsigned long long seed = dis(gen);

  AT_DISPATCH_FLOATING_TYPES(m.type(), "quantize_int8_cuda_kernel", ([&] {
    quantize_int8_cuda_kernel<scalar_t><<<blocks, threads0, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        m.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        (float *)scale.data_ptr(),
        q.packed_accessor32<int8_t,2,torch::RestrictPtrTraits>(),
        seed);
  }));

  m = m.reshape(m_shape);
  q = q.reshape(m_shape);

  return {q, scale};
}