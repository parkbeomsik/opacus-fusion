from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_rnn',
    ext_modules=[
        CUDAExtension('custom_rnn', [
            'custom_rnn.cpp',
            'lstm_cuda_kernel.cu'
        ], 
        libraries=["cublas"],
        include_dirs=["/home/beomsik/cuda-11.0/include",
                      ],
        library_dirs=["./"],
        extra_compile_args={'cxx': ["-O3", "-g"], # -g
                            'nvcc': ["-O3"]},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })