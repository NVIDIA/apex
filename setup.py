import torch
from setuptools import setup, find_packages

import sys

if not torch.cuda.is_available():
    print("Warning: Torch did not find available GPUs on this system.\n",
          "If your intention is to cross-compile, this is not an error.")

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("APEx requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

cmdclass = {}
ext_modules = []

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass['build_ext'] = BuildExtension

if "--cpp_ext" in sys.argv:
    from torch.utils.cpp_extension import CppExtension
    sys.argv.remove("--cpp_ext")
    ext_modules.append(
        CppExtension('apex_C',
                     ['csrc/flatten_unflatten.cpp',]))

if "--cuda_ext" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension
    sys.argv.remove("--cuda_ext")

    if torch.utils.cpp_extension.CUDA_HOME is None:
        print("Warning:  nvcc is not available.  Ignoring --cuda-ext") 
    else:
        ext_modules.append(
            CUDAExtension(name='amp_C',
                          sources=['csrc/scale_check_overflow.cpp',
                                   'csrc/scale_check_overflow_kernel.cu']))
        ext_modules.append(
            CUDAExtension(name='fused_adam_cuda',
                          sources=['apex/optimizers/csrc/fused_adam_cuda.cpp',
                                   'apex/optimizers/csrc/fused_adam_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3',],
                                              'nvcc':['-O3', 
                                                      '--use_fast_math']}))
        ext_modules.append(
            CUDAExtension(name='syncbn',
                          sources=['csrc/syncbn.cpp',
                                   'csrc/welford.cu']))
        ext_modules.append(
            CUDAExtension(name='fused_layer_norm_cuda',
                          sources=['apex/normalization/csrc/layer_norm_cuda.cpp',
                                   'apex/normalization/csrc/layer_norm_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3',],
                                              'nvcc':['-maxrregcount=50',
                                                      '-O3', 
                                                      '--use_fast_math']}))

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
    description='PyTorch Extensions written by NVIDIA',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
