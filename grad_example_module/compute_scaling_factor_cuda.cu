#include <cuda_runtime.h>

__global__ void compute_scaling_factor_cuda_kernel(float *out, 
                                                   const float *norm,
                                                   float max_norm) {
  *out = min((max_norm / (*norm + 1e-6)), 1.0);
}

void compute_scaling_factor_cuda(float *out, 
                                const float *norm,
                                float max_norm) {

    compute_scaling_factor_cuda_kernel<<<1, 1>>>(out, norm, max_norm);
}

__global__ void compute_scaling_factor2_cuda_kernel(float *out, 
                                                   const float *norm,
                                                   const float *norm2,
                                                   float max_norm) {
  *out = min((max_norm / (sqrt(powf(*norm, 2) + powf(*norm2, 2)) + 1e-6)), 1.0);
}

void compute_scaling_factor2_cuda(float *out, 
                                const float *norm,
                                const float *norm2,
                                float max_norm) {

    compute_scaling_factor2_cuda_kernel<<<1, 1>>>(out, norm, norm2, max_norm);
}