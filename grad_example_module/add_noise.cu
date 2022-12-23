#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstdlib>

#include "structure.h"
#include "error_helper.h"
#include "utils.h"

#include <curand_kernel.h>

bool _add_noise_first_run = true;

std::mt19937 noise_gen(1234);
std::uniform_int_distribution<unsigned long long> noise_dis(0, ULLONG_MAX);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void add_noise_cuda_kernel(
    float* m,
    int64_t num_elem,
    float noise_dev,
    unsigned long long seed) {
  // Column index
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  while (c < num_elem){
    curandState s;
    curand_init(seed + c, 0, 0, &s);
    m[c] += curand_normal(&s) * noise_dev;

    c += blockDim.x * gridDim.x;
  }
}

void add_noise(torch::Tensor& tensor, float noise_dev) {
  if (_add_noise_first_run) {

  }

    const int threads = 1024;
    const dim3 blocks(std::min((tensor.numel() + threads - 1) / threads, (int64_t)360), 1);

    std::mt19937 noise_gen((unsigned)std::time(NULL));
    unsigned long long seed = noise_dis(noise_gen);
    // printf("%f\n", noise_dev);
    
    add_noise_cuda_kernel<<<blocks, threads, 0>>>(
        (float *) tensor.data_ptr(),
        tensor.numel(),
        noise_dev,
        seed);

}