from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='grad_example_module',
    ext_modules=[
        CUDAExtension('grad_example_module', [
            'grad_example_module_conv.cpp',
            'grad_example_module_linear.cpp',
            'compute_scaling_factor_cuda.cu',
            'grad_example_module.cpp',
            'quantize.cu',
            'cutlass_simt_int8_batched_gemm.cu'
        ], 
        libraries=["cudnn", "cutlass_wgrad_grouped",],
        include_dirs=["/home/beomsik/cuda-11.0/include",
                      "/home/beomsik/dp/opacus-fusion/cutlass_wgrad_grouped/build/include",
                      "/home/beomsik/dp/cutlass/include"
                      ],
        library_dirs=["./", "/home/beomsik/dp/opacus-fusion/cutlass_wgrad_grouped/build/lib"],
        extra_compile_args={'cxx': ["-O3", "-g"], # -g
                            'nvcc': ["-O3"]},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })