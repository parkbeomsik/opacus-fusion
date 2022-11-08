#pragma once

#include <vector>

#include <vector>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cuda_runtime.h"

#include "../error_helper.h"

#include "operation.h"

template< 
  typename ThreadblockShape,
  typename MmaShape,
  typename InstructionShape>
class IgemmBatchedSimt : public GemmBatchedOperation {
public:
  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    ThreadblockShape,
    MmaShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      float, 
      1,
      int32_t, 
      float
    >
  >;

public:

  Gemm gemm_op;

  IgemmBatchedSimt(std::string name):
    GemmBatchedOperation(name) {};

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
                  cudaStream_t stream) {

    typename Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                        A, cutlass::layout::RowMajor(lda),
                        B, cutlass::layout::ColumnMajor(ldb),
                        C, cutlass::layout::ColumnMajor(ldc),
                        C, cutlass::layout::ColumnMajor(ldc),
                        typename Gemm::EpilogueOutputOp::Params(alpha, beta),
                        batch_count);

    cutlass::Status status = gemm_op(args, nullptr, stream);

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }

    return cudaSuccess;
  }
};
