from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import argparse

import torch
import os

# parser = argparse.ArgumentParser(description="Install grad_example_module")
# parser.add_argument(
#     "--cutlass_path", type=str, required=True, help="Path of cutlass"
# )
# parser.add_argument(
#     "--cutlass_wgrad_grouped_path", type=str, required=True, help="Path of cutlass_wgrad_grouped"
# )
# args = parser.parse_args()

if "A100" in torch.cuda.get_device_name():
    is_sm_80 = True
else:
    is_sm_80 = False

gencode = ["-gencode=arch=compute_80,code=compute_80"] if is_sm_80 else []
cxx_flags = ["-D_USE_TENSOR_CORE"] if is_sm_80 else []
# cxx_flags = []

os.system(f"nvcc -Xcompiler -fPIC -std=c++14  -I/home/beomsik/dp/opacus-fusion/cutlass_wgrad_grouped/build/include -I/home/beomsik/dp/cutlass/include cutlass_simt_int8_wgrad.cu -c -o libcutlass_simt_int8_wgrad.o")
os.system("ar rcs libcutlass_simt_int8_wgrad.a libcutlass_simt_int8_wgrad.o")

setup(
    name='grad_example_module',
    ext_modules=[
        CUDAExtension('grad_example_module', [
            'grad_example_module_conv.cpp',
            'grad_example_module_linear.cpp',
            'compute_scaling_factor_cuda.cu',
            'grad_example_module.cpp',
            'quantize.cu',
            'add_noise.cu',
            'cutlass_simt_int8_batched_gemm.cu',
            # 'cutlass_simt_int8_wgrad.cu'
        ], 
        libraries=["cudnn", "cutlass_wgrad_grouped", "cutlass_simt_int8_wgrad"],
        include_dirs=[#"/home/beomsik/cuda-11.0/include",
                      f"/home/beomsik/dp/opacus-fusion/cutlass_wgrad_grouped/build/include",
                      f"/home/beomsik/dp/cutlass/include"
                      ],
        library_dirs=["./", f"/home/beomsik/dp/opacus-fusion/cutlass_wgrad_grouped/build/lib"],
        extra_compile_args={'cxx': cxx_flags, # -g
                            'nvcc': gencode},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })