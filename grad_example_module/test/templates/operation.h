#pragma once

#include <string>

#include "cuda_runtime.h"

// Base class for all operations
class GemmBatchedOperation {
public:
  std::string name;

  GemmBatchedOperation(std::string name) {this->name = name;};
  virtual ~GemmBatchedOperation() { }

  virtual cudaError_t run(
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
                cudaStream_t stream) = 0;

};

// Base class for Conv operations
class Operation {
public:
  std::string name;

  Operation(std::string name) {this->name = name;};
  virtual ~Operation() { }

  virtual cudaError_t run(int8_t * ograd,
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
                  int split_k_slices,
                  float alpha,
                  float beta,
                  cudaStream_t stream) = 0;

  virtual size_t get_workspace(
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
    int split_k_slices) = 0;
};