from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_sampling',
    ext_modules=[
        CUDAExtension(
            name='_backend', # 'mmdet.ops.roi_sampling._backend',
            sources=[
                'src/roi_sampling.cpp',
                'src/roi_sampling_cpu.cpp',
                'src/roi_sampling_cuda.cu',
            ],
            include_dirs=[
                'src',          # roi_sampling.h
                'src/utils',    # checks.h, common.h, cuda.cuh
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
