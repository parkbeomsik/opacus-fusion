import os
import sys
from typing import Tuple
import time
from tqdm import tqdm
import time
import subprocess
import traceback
import argparse
import datetime
import torch
import json
import itertools
from datetime import timedelta

import generate_initialize_all_cpp
import generate_initialize_all_h
import generate_kernel
import generate_test_kernel

def main():
    cutlass_dir = "/home/beomsik/dp/cutlass"
    test_source_file_name = "kernel_test.cu"
    test_excutable_name = "./kernel_test"
    compile_command = f"nvcc -I{cutlass_dir}/include -I{cutlass_dir}/tools/util/include \
                        {test_source_file_name} -o {test_excutable_name}"
    print(compile_command)

    dtype = "int"

    # Sweep parameters
    tb_shape_0_cand = [64, 128, 256]
    tb_shape_1_cand = [64, 128, 256]
    tb_shape_2_cand = [2, 4, 8, 32]
    mma_shape_0_cand = [16, 32, 64, 128]
    mma_shape_1_cand = [16, 32, 64, 128]
    mma_shape_2_cand = [2, 4, 8, 32]
    instruction_shape_0_cand = [1]
    instruction_shape_1_cand = [1]
    instruction_shape_2_cand = [4]
    num_stages_cand = [3]

    # tb_shape_0_cand = [32]
    # tb_shape_1_cand = [32]
    # tb_shape_2_cand = [16]
    # mma_shape_0_cand = [16]
    # mma_shape_1_cand = [16]
    # mma_shape_2_cand = [16]
    # instruction_shape_0_cand = [1]
    # instruction_shape_1_cand = [1]
    # instruction_shape_2_cand = [1, 4]
    # num_stages_cand = [4]

    all_cand = list(itertools.product(tb_shape_0_cand, tb_shape_1_cand, tb_shape_2_cand,
                                 mma_shape_0_cand, mma_shape_1_cand, mma_shape_2_cand,
                                 instruction_shape_0_cand, instruction_shape_1_cand, instruction_shape_2_cand,
                                 num_stages_cand))

    compile_results = {"results": {}}

    log_f_name = f"compile_results_{dtype}.json"
    if os.path.exists(log_f_name):
        is_exist = False
        with open(log_f_name, "r") as f:
            if f.readlines():
                is_exist = True
        if is_exist:
            with open(log_f_name, "r") as f:
                compile_results = json.load(f)

    # success_log_f = open(f"results/{model_name}_{dataset}_{quant_mode}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_result_success_log.txt", "w")

    # Compile test and write kernle .cu files
    cur_path = os.getcwd()
    start_time = time.time()
    num_complted_cases = 0
    for (tb_0, tb_1, tb_2,
         mma_0, mma_1, mma_2,
         inst_0, inst_1, inst_2,
         num_stages) in tqdm(all_cand):
        num_complted_cases += 1          

        kernel_name = generate_kernel.get_kernel_name(dtype, "OpClassSimt", (tb_0, tb_1, tb_2), (mma_0, mma_1, mma_2), (inst_0, inst_1, inst_2), num_stages)           

        try:                                
            if kernel_name in compile_results["results"].keys(): 
                result = compile_results["results"][kernel_name]
                print(result)
                if result == "Failed":
                    raise RuntimeError
            
            else:
                result=None
                if (tb_0 < mma_0) or (tb_1 < mma_1) or (tb_2 < mma_2) or \
                    (mma_0 < inst_0) or (mma_1 < inst_1):
                    result="Failed"
                    raise RuntimeError
                
                if (tb_2 != mma_2):
                    result="Failed"
                    raise RuntimeError

                with open(test_source_file_name, "w") as f:
                    f.write(generate_test_kernel.get_string(dtype, "OpClassSimt", (tb_0, tb_1, tb_2), (mma_0, mma_1, mma_2), (inst_0, inst_1, inst_2), num_stages))

                completed_process = subprocess.run(compile_command.split())
                if completed_process.returncode != 0:
                    tqdm.write("Build failed")
                    result="Failed"
                    raise RuntimeError

            result = "Success"
            with open(f"src/kernel_{dtype}/{kernel_name}.cu", "w") as f:
                f.write(generate_kernel.get_string(dtype, "OpClassSimt", (tb_0, tb_1, tb_2), (mma_0, mma_1, mma_2), (inst_0, inst_1, inst_2), num_stages))

        except KeyboardInterrupt:
            sys.exit()

        except:
            print(traceback.format_exc())
            result="Failed"
            pass
        
        finally:
            os.chdir(cur_path)
            tqdm.write(f"{result} ms")

            compile_results["results"][kernel_name] = result

            with open(log_f_name, "w") as log_f:
                json.dump(compile_results, log_f, indent=4, sort_keys=True)

    success_kernel_names = []
    for kernel_name in compile_results["results"]:
        if compile_results["results"][kernel_name] == "Success":
            success_kernel_names.append(kernel_name)

    with open(f"src/initialize_{dtype}.h", "w") as f:
        f.write(generate_initialize_all_h.get_string(success_kernel_names))

    with open(f"src/initialize_{dtype}.cpp", "w") as f:
        f.write(generate_initialize_all_cpp.get_string(success_kernel_names))


    print("=====================  Experiment results =======================\n")
    print(f"Time elapsed {timedelta(seconds=time.time() - start_time)}")
    print(f"Write {len(success_kernel_names)} kernel .cu files")

if __name__ == "__main__":
    main()
