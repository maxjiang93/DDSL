from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dsnet_cuda',
    ext_modules=[
        CUDAExtension('dsnet_cuda', [
            'dsnet_cuda.cpp',
            'denet_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
