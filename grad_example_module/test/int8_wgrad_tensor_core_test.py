import subprocess
import itertools
import os

from tqdm import tqdm

def main():
  tb_shape_0_cand = [32, 64, 128, 256]
  tb_shape_1_cand = [32, 64, 128, 256]
  tb_shape_2_cand = [32, 64, 128]
  mma_shape_0_cand = [32, 64, 128, 256]
  mma_shape_1_cand = [32, 64, 128, 256]
  mma_shape_2_cand = [32, 64, 128]
  inst_shape_0_cand = [8, 16, 32]
  inst_shape_1_cand = [8, 16, 32]
  inst_shape_2_cand = [8, 16, 32]
  split_k_slice_cand = [2, 4, 8, 16, 32, 64]
  # split_k_slice_cand = [2]

  # tb_shape_0_cand = [128]
  # tb_shape_1_cand = [128]
  # tb_shape_2_cand = [64]
  # mma_shape_0_cand = [64]
  # mma_shape_1_cand = [64]
  # mma_shape_2_cand = [64]
  # inst_shape_0_cand = [16]
  # inst_shape_1_cand = [8]
  # inst_shape_2_cand = [32]

  # problem = [N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w]
  problems = [

              [1, 32, 32, 3, 64, 7, 7, 16, 16, 3, 3, 2, 2, 1, 1],
              [1, 224, 224, 3, 64, 7, 7, 112, 112, 3, 3, 2, 2, 1, 1],

              [1, 56, 56, 64, 64, 1, 1, 56, 56, 0, 0, 1, 1, 1, 1],
              [1, 56, 56, 64, 64, 3, 3, 56, 56, 1, 1, 1, 1, 1, 1],
              [1, 56, 56, 64, 256, 1, 1, 56, 56, 1, 1, 1, 1, 1, 1],
              [1, 56, 56, 256, 64, 1, 1, 56, 56, 1, 1, 1, 1, 1, 1]
              ]

  for problem in problems:

    all_cases = itertools.product(tb_shape_0_cand, tb_shape_1_cand, tb_shape_2_cand,
                                  mma_shape_0_cand, mma_shape_1_cand, mma_shape_2_cand,
                                  inst_shape_0_cand, inst_shape_1_cand, inst_shape_2_cand,
                                  split_k_slice_cand)
    num_cases = len(tb_shape_0_cand) * len(tb_shape_1_cand) * len(tb_shape_2_cand) *\
                len(mma_shape_0_cand) * len(mma_shape_1_cand) * len(mma_shape_2_cand) *\
                len(inst_shape_0_cand) * len(inst_shape_1_cand) * len(inst_shape_2_cand) *\
                len(split_k_slice_cand)

    min_runtime = 1000.0
    min_conf = None

    for conf in tqdm(all_cases, total=num_cases):
      tb_shape = [conf[0], conf[1], conf[2]]
      mma_shape = [conf[3], conf[4], conf[5]]
      inst_shape = [conf[6], conf[7], conf[8]]
      split_k_slice = conf[9]

      if mma_shape[0] > tb_shape[0] or mma_shape[1] > tb_shape[1] or mma_shape[2] > tb_shape[2]:
        continue

      exec_name = f'int8_wgrad_tensor_core_kernels/tensorop_iwgrad_{tb_shape[0]}x{tb_shape[1]}x{tb_shape[2]}_{mma_shape[0]}x{mma_shape[1]}x{mma_shape[2]}_{inst_shape[0]}x{inst_shape[1]}x{inst_shape[2]}_splitK_test'

      if not os.path.isfile(exec_name):       
        continue

      ret = subprocess.run(f"tasekset -c 0-20 {exec_name} {' '.join(map(lambda x: str(x), problem))} {split_k_slice}", shell=True, capture_output=True)
      # print(ret.stderr)

      # print(f"{tb_shape}, {mma_shape}, {inst_shape} : {ret.stdout.decode()[:-1]} ms")

      # subprocess.run(exec_name, shell=True, capture_output=True)

      try:
        if float(ret.stdout.decode()[:-1]) < min_runtime:
          min_runtime = float(ret.stdout.decode()[:-1])
          min_conf = conf
      except:
        pass

    print(f"{problem} : <{min_conf[0]}, {min_conf[1]}, {min_conf[2]}>, <{min_conf[3]}, {min_conf[4]}, {min_conf[5]}>, <{min_conf[6]}, {min_conf[7]}, {min_conf[8]}>, {min_conf[9]}, {min_runtime} ms")


if __name__ == "__main__":
  main()