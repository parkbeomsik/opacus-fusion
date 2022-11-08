import subprocess

def main():
  # problem = [N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w]
  exec_name = "./test_iwgrad_simt"
  problems = [

              [1, 32, 32, 3, 64, 7, 7, 16, 16, 3, 3, 2, 2, 1, 1],
              [1, 224, 224, 3, 64, 7, 7, 112, 112, 3, 3, 2, 2, 1, 1],

              [1, 56, 56, 64, 64, 1, 1, 56, 56, 0, 0, 1, 1, 1, 1],
              [1, 56, 56, 64, 64, 3, 3, 56, 56, 1, 1, 1, 1, 1, 1],
              [1, 56, 56, 64, 256, 1, 1, 56, 56, 1, 1, 1, 1, 1, 1],
              [1, 56, 56, 256, 64, 1, 1, 56, 56, 1, 1, 1, 1, 1, 1]
              ]

  for problem in problems:

    ret = subprocess.run(f"taskset -c 0-20 {exec_name} {' '.join(map(lambda x: str(x), problem))}", shell=True)
    # print(ret.stderr)

    # print(f"{tb_shape}, {mma_shape}, {inst_shape} : {ret.stdout.decode()[:-1]} ms")

    # subprocess.run(exec_name, shell=True, capture_output=True)


if __name__ == "__main__":
  main()