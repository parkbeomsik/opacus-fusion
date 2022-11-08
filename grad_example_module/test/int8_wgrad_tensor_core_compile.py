import subprocess
import itertools
import os
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
from multiprocessing import Pool

from tqdm import tqdm

def get_source_code(problem, tb_shape, mma_shape, inst_shape, split_k_slice):
  conf_str = f"{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}"

  return f'''
#include "../templates/conv2d_wgrad_tensorop.h"

void initialize_cutlass_tensorop_iwgrad_grouped_{conf_str}(
  std::vector<Operation *>& ops) {{
 
  ops.push_back(new Conv2dWgradTensorOp<
  cutlass::gemm::GemmShape<{tb_shape[0]}, {tb_shape[1]}, {tb_shape[2]}>,
  cutlass::gemm::GemmShape<{mma_shape[0]}, {mma_shape[1]}, {mma_shape[2]}>,
  cutlass::gemm::GemmShape<{inst_shape[0]}, {inst_shape[1]}, {inst_shape[2]}>>(
      "cutlass_tensorop_iwgrad_grouped_{conf_str}"));
}}
'''

def get_header(conf_list):

  declares = ""
  for conf in conf_list:
    declares += f"void initialize_cutlass_tensorop_iwgrad_grouped_{conf}(std::vector<Operation *>& ops);\n"

  inits = ""
  for conf in conf_list:
    inits += f"  initialize_cutlass_tensorop_iwgrad_grouped_{conf}(ops);\n"

  return f'''
#pragma once

#include "../templates/conv2d_wgrad_tensorop.h"
#include "../templates/operation.h"

{declares}

void initialize_conv2d_wgrad_tensorop (std::vector<Operation *>& ops) {{
{inits}
}}
'''

def compile(conf):
  tb_shape = [conf[0], conf[1], conf[2]]
  mma_shape = [conf[3], conf[4], conf[5]]
  inst_shape = [conf[6], conf[7], conf[8]]

  if mma_shape[0] > tb_shape[0] or mma_shape[1] > tb_shape[1] or mma_shape[2] > tb_shape[2]:
    return ""

  exec_name = f'tensorop_iwgrad_kernels/tensorop_iwgrad_{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}_splitK.o'

  if not os.path.isfile(exec_name):       
    # continue

    with open(f"test_kernels/test_wgrad_main_{mp.current_process().name}.cu", "w") as f:
      f.write(get_source_code(None, tb_shape, mma_shape, inst_shape, None))

    compile_ret = subprocess.run(f"nvcc -gencode=arch=compute_80,code=compute_80 -Xcompiler -fPIC -std=c++14 -I/home/beomsik/dp/cutlass/include test_kernels/test_wgrad_main_{mp.current_process().name}.cu -c -o {exec_name}", shell=True, capture_output=True)

    # print(compile_ret.stderr)

    if compile_ret.returncode == 0:
      return f"{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}"
    if compile_ret.returncode != 0:
      return ""
  

def main():
  tb_shape_0_cand = [32, 64, 128, 256]
  tb_shape_1_cand = [32, 64, 128, 256]
  tb_shape_2_cand = [32, 64, 128, 256]
  mma_shape_0_cand = [32, 64, 128]
  mma_shape_1_cand = [32, 64, 128]
  mma_shape_2_cand = [32, 64, 128]
  inst_shape_0_cand = [8, 16, 32]
  inst_shape_1_cand = [8, 16, 32]
  inst_shape_2_cand = [8, 16, 32]

  # tb_shape_0_cand = [64, 128]
  # tb_shape_1_cand = [64, 128]
  # tb_shape_2_cand = [64, 128]
  # mma_shape_0_cand = [64, 128]
  # mma_shape_1_cand = [64, 128]
  # mma_shape_2_cand = [64, 128]
  # inst_shape_0_cand = [8, 16, 32]
  # inst_shape_1_cand = [8, 16, 32]
  # inst_shape_2_cand = [8, 16, 32]

  # tb_shape_0_cand = [128]
  # tb_shape_1_cand = [128]
  # tb_shape_2_cand = [64]
  # mma_shape_0_cand = [64]
  # mma_shape_1_cand = [64]
  # mma_shape_2_cand = [64]
  # inst_shape_0_cand = [16]
  # inst_shape_1_cand = [8]
  # inst_shape_2_cand = [32]

  all_cases = itertools.product(tb_shape_0_cand, tb_shape_1_cand, tb_shape_2_cand,
                                mma_shape_0_cand, mma_shape_1_cand, mma_shape_2_cand,
                                inst_shape_0_cand, inst_shape_1_cand, inst_shape_2_cand)
  num_cases = len(tb_shape_0_cand) * len(tb_shape_1_cand) * len(tb_shape_2_cand) *\
              len(mma_shape_0_cand) * len(mma_shape_1_cand) * len(mma_shape_2_cand) *\
              len(inst_shape_0_cand) * len(inst_shape_1_cand) * len(inst_shape_2_cand)

  with Pool(128) as p:
    # compile each case, return 1 if success else 0
    ret = list(tqdm(p.imap(compile, all_cases)))

    print(f"{sum(map(lambda x: 0 if x == '' else 1, ret))} kernels copmiled.")

    p.close()
    p.join()

  success_confs = [f.split("_", 2)[2][:-9] for f in 
                    listdir("tensorop_iwgrad_kernels")
                    if (isfile(join("tensorop_iwgrad_kernels", f) )
                      and f[-1] == "o")]

  with open(f'tensorop_iwgrad_kernels/initialize_tensorop_iwgrad_splitK.h', "w") as f:
    f.write(get_header(success_confs))



if __name__ == "__main__":
  main()