import subprocess
import itertools
import os

from tqdm import tqdm

def get_source_code(m, n, k, batch_count, tb_shape, mma_shape, inst_shape):
  return f'''
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
  cudaStream_t stream) {{  

  using namespace cutlass::gemm::device;

  using Gemm = cutlass::gemm::device::GemmArray<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<{tb_shape[0]}, {tb_shape[1]}, {tb_shape[2]}>,
    cutlass::gemm::GemmShape<{mma_shape[0]}, {mma_shape[1]}, {mma_shape[2]}>,
    cutlass::gemm::GemmShape<{inst_shape[0]}, {inst_shape[1]}, {inst_shape[2]}>,
    cutlass::epilogue::thread::LinearCombination<
      float, 
      1,
      int32_t, 
      float
    >
  >;

  Gemm gemm_op;

  Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k),
                       A, cutlass::layout::RowMajor(lda),
                       B, cutlass::layout::ColumnMajor(ldb),
                       C, cutlass::layout::ColumnMajor(ldc),
                       C, cutlass::layout::ColumnMajor(ldc),
                       Gemm::EpilogueOutputOp::Params(alpha, beta),
                       batch_count);

  cutlass::Status status = gemm_op(args, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {{
    return cudaErrorUnknown;
  }}

  return cudaSuccess;
}}

int main(int argc, char * argv[]) {{
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int k = atoi(argv[3]);
  int batch_count = atoi(argv[4]);

  std::vector<void *> host_A_array(batch_count, NULL);
  std::vector<void *> host_B_array(batch_count, NULL);
  std::vector<void *> host_C_array(batch_count, NULL);

  for (int i = 0; i < batch_count; ++i) {{
    checkCudaErrors(cudaMalloc(&host_A_array[i], sizeof(int8_t)*m*k));
    checkCudaErrors(cudaMalloc(&host_B_array[i], sizeof(int8_t)*k*n));
    checkCudaErrors(cudaMalloc(&host_C_array[i], sizeof(float)*m*n));
  }}
  
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
  for (int i = 0; i < 3; ++i) {{
    cutlass_simt_igemm_int8_batched_gemm(m, n, k,
                                         1.0,
                                         (int8_t **)device_A_array,
                                         k,
                                         (int8_t **)device_B_array,
                                         k,
                                         (float **)device_C_array,
                                         m,
                                         0.0,
                                         batch_count,
                                         NULL);
  }}

  cudaEvent_t events[2];
  checkCudaErrors(cudaEventCreate(&events[0]));
  checkCudaErrors(cudaEventCreate(&events[1]));

  checkCudaErrors(cudaEventRecord(events[0]));

  // Measure runtime_ms
  for (int i = 0; i < 20; ++i) {{
    checkCudaErrors(cutlass_simt_igemm_int8_batched_gemm(m, n, k,
                                                        1.0,
                                                        (int8_t **)device_A_array,
                                                        k,
                                                        (int8_t **)device_B_array,
                                                        k,
                                                        (float **)device_C_array,
                                                        m,
                                                        0.0,
                                                        batch_count,
                                                        NULL));
  }}

  checkCudaErrors(cudaEventRecord(events[1]));
  checkCudaErrors(cudaDeviceSynchronize());

  float runtime_ms = 0.0;
  checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

  std::cout << runtime_ms / 20.0 << std::endl;
}}
'''

def main():
  tb_shape_0_cand = [128, 256]
  tb_shape_1_cand = [128, 256]
  tb_shape_2_cand = [4, 8, 16, 32, 64, 128]
  mma_shape_0_cand = [32, 64, 128, 256]
  mma_shape_1_cand = [32, 64, 128, 256]
  mma_shape_2_cand = [4, 8, 16, 32, 64, 128]
  inst_shape_0_cand = [1]
  inst_shape_1_cand = [1]
  inst_shape_2_cand = [4]

  # tb_shape_0_cand = [128]
  # tb_shape_1_cand = [128]
  # tb_shape_2_cand = [32]
  # mma_shape_0_cand = [64]
  # mma_shape_1_cand = [64]
  # mma_shape_2_cand = [32]
  # inst_shape_0_cand = [1]
  # inst_shape_1_cand = [1]
  # inst_shape_2_cand = [4]

  # problem = [m, n, k, batch_count]
  problems = [
              [768, 768, 32, 48],
              [3072, 768, 32, 12],
              [768, 3072, 32, 12],
              [768, 768, 256, 48],
              [3072, 768, 256, 12],
              [768, 3072, 256, 12],
              [1024, 1024, 32, 96],
              [4096, 1024, 32, 24],
              [1024, 4096, 32, 24],
              [1024, 1024, 256, 96],
              [4096, 1024, 256, 24],
              [1024, 4096, 256, 24],
              ]

  for problem in problems:
    m = problem[0]
    n = problem[1]
    k = problem[2]
    batch_count = problem[3]
    

    all_cases = itertools.product(tb_shape_0_cand, tb_shape_1_cand, tb_shape_2_cand,
                                  mma_shape_0_cand, mma_shape_1_cand, mma_shape_2_cand,
                                  inst_shape_0_cand, inst_shape_1_cand, inst_shape_2_cand)
    num_cases = len(tb_shape_0_cand) * len(tb_shape_1_cand) * len(tb_shape_2_cand) *\
                len(mma_shape_0_cand) * len(mma_shape_1_cand) * len(mma_shape_2_cand) *\
                len(inst_shape_0_cand) * len(inst_shape_1_cand) * len(inst_shape_2_cand)

    min_runtime = 1000.0
    min_conf = None

    for conf in tqdm(all_cases, total=num_cases):
      tb_shape = [conf[0], conf[1], conf[2]]
      mma_shape = [conf[3], conf[4], conf[5]]
      inst_shape = [conf[6], conf[7], conf[8]]

      if mma_shape[0] > tb_shape[0] or mma_shape[1] > tb_shape[1] or mma_shape[2] > tb_shape[2]:
        continue
      
      exec_name = f'int8_batched_gemm_kernels/simt_igemm_batched_{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}_test'

      if not os.path.isfile(exec_name):
        continue
        with open("test_main.cu", "w") as f:
          f.write(get_source_code(m, n, k, batch_count, tb_shape, mma_shape, inst_shape))

        compile_ret = subprocess.run(f"nvcc -I/home/beomsik/dp/cutlass/include test_main.cu -o {exec_name}", shell=True, capture_output=True)
        if compile_ret.returncode != 0:
          # print(f"{tb_shape}, {mma_shape}, {inst_shape} : error")
          continue

      ret = subprocess.run(f"{exec_name} {' '.join(map(lambda x: str(x), problem))}", shell=True, capture_output=True)

      # print(f"{tb_shape}, {mma_shape}, {inst_shape} : {ret.stdout.decode()[:-1]} ms")
      try:
        if float(ret.stdout.decode()[:-1]) < min_runtime:
          min_runtime = float(ret.stdout.decode()[:-1])
          min_conf = conf
      except:
        pass

    # print(min_conf)
    # print(min_runtime)
    print(f"{problem[3]}x{problem[0]}x{problem[1]}x{problem[2]} : <{min_conf[0]}, {min_conf[1]}, {min_conf[2]}>, <{min_conf[3]}, {min_conf[4]}, {min_conf[5]}>, <{min_conf[6]}, {min_conf[7]}, {min_conf[8]}>, {min_runtime} ms")


if __name__ == "__main__":
  main()