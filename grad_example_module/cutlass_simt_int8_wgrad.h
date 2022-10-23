#include <vector>

#include "cuda_runtime.h"

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
  float alpha = 1.0,
  float beta = 0.0,
  cudaStream_t stream = NULL);

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
  int dilation_w);