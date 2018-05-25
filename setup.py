import torch.cuda
from setuptools import setup, find_packages
from distutils.command.clean import clean
from torch.utils.cpp_extension import CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

# TODO:  multiple modules, so we don't have to route all interfaces through
# the same interface.cpp file?
if torch.cuda.is_available() and CUDA_HOME is not None:
    ext_modules = []
    extension = CUDAExtension(
        'apex._C', [
            'csrc/interface.cpp',
            'csrc/weight_norm_fwd_cuda.cu',
            'csrc/weight_norm_bwd_cuda.cu',
            'csrc/scale_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2', '-arch=sm_70']}) # TODO:  compile for all arches.
    ext_modules.append(extension)
else:
    raise RuntimeError("Apex requires Cuda 9.0 or higher")

setup(
    name='apex',
    version='0.1',
    packages=find_packages(exclude=('build', 
                                    'csrc', 
                                    'include', 
                                    'tests', 
                                    'dist',
                                    'docs',
                                    'tests',
                                    'examples',
                                    'apex.egg-info',)),
    ext_modules=ext_modules,
    description='PyTorch Extensions written by NVIDIA',
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)
