import subprocess
import itertools
import os
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
from multiprocessing import Pool

from tqdm import tqdm

def get_source_code(tb_shape, mma_shape, inst_shape):
  conf_str = f"{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}"

  return f'''
#include "../templates/igemm_batched_simt.h"

void initialize_cutlass_simt_igemm_batched_{conf_str}(
  std::vector<GemmBatchedOperation *>& ops) {{
 
  ops.push_back(new IgemmBatchedSimt<
  cutlass::gemm::GemmShape<{tb_shape[0]}, {tb_shape[1]}, {tb_shape[2]}>,
  cutlass::gemm::GemmShape<{mma_shape[0]}, {mma_shape[1]}, {mma_shape[2]}>,
  cutlass::gemm::GemmShape<{inst_shape[0]}, {inst_shape[1]}, {inst_shape[2]}>>(
      "cutlass_simt_igemm_batched_{conf_str}"));
}}
'''

def get_header(conf_list):

  declares = ""
  for conf in conf_list:
    declares += f"void initialize_cutlass_simt_igemm_batched_{conf}(std::vector<GemmBatchedOperation *>& ops);\n"

  inits = ""
  for conf in conf_list:
    inits += f"  initialize_cutlass_simt_igemm_batched_{conf}(ops);\n"

  return f'''
#pragma once

#include "../templates/igemm_batched_simt.h"
#include "../templates/operation.h"

{declares}

void initialize_igemm_batched_simt (std::vector<GemmBatchedOperation *>& ops) {{
{inits}
}}
'''

def compile(conf):
  tb_shape = [conf[0], conf[1], conf[2]]
  mma_shape = [conf[3], conf[4], conf[5]]
  inst_shape = [conf[6], conf[7], conf[8]]

  if mma_shape[0] > tb_shape[0] or mma_shape[1] > tb_shape[1] or mma_shape[2] > tb_shape[2]:
    return ""

  exec_name = f'_simt_igemm_batched_kernels/simt_igemm_batched_{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}.o'

  if not os.path.isfile(exec_name):       
    # continue

    with open(f"_test_kernels/test_igemm_batched_main_{mp.current_process().name}.cu", "w") as f:
      f.write(get_source_code(tb_shape, mma_shape, inst_shape))

    compile_ret = subprocess.run(f"nvcc -gencode=arch=compute_70,code=compute_70 -Xcompiler -fPIC -std=c++14 -I/home/beomsik/dp/cutlass/include _test_kernels/test_igemm_batched_main_{mp.current_process().name}.cu -c -o {exec_name}", shell=True, capture_output=True)

    # print(compile_ret.stderr.decode())

    if compile_ret.returncode == 0:
      return f"{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}"
    if compile_ret.returncode != 0:
      return ""

def main():
  if not os.path.exists("_simt_igemm_batched_kernels"):
    os.mkdir("_simt_igemm_batched_kernels")
  if not os.path.exists("_test_kernels"):
    os.mkdir("_test_kernels")

  tb_shape_0_cand = [128, 256, 512]
  tb_shape_1_cand = [128, 256, 512]
  tb_shape_2_cand = [1, 4, 8, 16, 32, 64, 128]
  mma_shape_0_cand = [32, 64, 128, 256, 512]
  mma_shape_1_cand = [32, 64, 128, 256, 512]
  mma_shape_2_cand = [1, 4, 8, 16, 32, 64, 128]
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
    
  all_cases = itertools.product(tb_shape_0_cand, tb_shape_1_cand, tb_shape_2_cand,
                                mma_shape_0_cand, mma_shape_1_cand, mma_shape_2_cand,
                                inst_shape_0_cand, inst_shape_1_cand, inst_shape_2_cand)

  num_cases = len(tb_shape_0_cand) * len(tb_shape_1_cand) * len(tb_shape_2_cand) *\
              len(mma_shape_0_cand) * len(mma_shape_1_cand) * len(mma_shape_2_cand) *\
              len(inst_shape_0_cand) * len(inst_shape_1_cand) * len(inst_shape_2_cand)

  with Pool(80) as p:
    # compile each case, return 1 if success else 0
    ret = list(tqdm(p.imap(compile, all_cases), total=num_cases))

    print(f"{sum(map(lambda x: 0 if x == '' else 1, ret))} kernels copmiled.")

    p.close()
    p.join()

  success_confs = [f.split("_", 2)[2][:-9] for f in 
                    listdir("_simt_igemm_batched_kernels")
                    if (isfile(join("_simt_igemm_batched_kernels", f) )
                      and f[-1] == "o")]

  with open(f'_simt_igemm_batched_kernels/initialize_simt_igemm_batched.h', "w") as f:
    f.write(get_header(success_confs))


if __name__ == "__main__":
  main()