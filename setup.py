import torch.cuda
import os
import re
import subprocess
from setuptools import setup, find_packages
from distutils.command.clean import clean
from torch.utils.cpp_extension import CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

# TODO:  multiple modules, so we don't have to route all interfaces through
# the same interface.cpp file?

if not torch.cuda.is_available():
    print("Warning: Torch did not find available GPUs on this system.\n",
          "If your intention is to cross-compile, this is not an error.")

def find(path, regex_func, collect=False):
    collection = [] if collect else None
    for root, dirs, files in os.walk(path):
        for file in files:
            if regex_func(file):
                if collect:
                    collection.append(os.path.join(root, file))
                else:
                    return os.path.join(root, file)
    return list(set(collection))

def get_cuda_version():
    NVCC = find(CUDA_HOME+os.sep+"bin",
                re.compile('nvcc$').search)
    print("Found NVCC = ", NVCC)

    # Parse output of nvcc to get cuda major version
    nvcc_output = subprocess.check_output([NVCC, '--version']).decode("utf-8")
    CUDA_LIB = re.compile(', V[0-9]+\.[0-9]+\.[0-9]+').search(nvcc_output).group(0).split('V')[1]
    print("Found CUDA_LIB = ", CUDA_LIB)

    CUDA_MAJOR_VERSION = int(CUDA_LIB.split('.')[0])
    print("Found CUDA_MAJOR_VERSION = ", CUDA_MAJOR_VERSION)

    if CUDA_MAJOR_VERSION < 8:
        raise RuntimeError("APex requires CUDA 8.0 or newer")

    return CUDA_MAJOR_VERSION

if CUDA_HOME is not None:
    print("Found CUDA_HOME = ", CUDA_HOME)

    CUDA_MAJOR_VERSION = get_cuda_version()

    gencodes = ['-gencode', 'arch=compute_52,code=sm_52',
                '-gencode', 'arch=compute_60,code=sm_60',
                '-gencode', 'arch=compute_61,code=sm_61',]

    if CUDA_MAJOR_VERSION > 8:
        gencodes += ['-gencode', 'arch=compute_70,code=sm_70',
                     '-gencode', 'arch=compute_70,code=compute_70',]

    ext_modules = []
    extension = CUDAExtension(
        'apex._C', [
            'csrc/interface.cpp',
            'csrc/weight_norm_fwd_cuda.cu',
            'csrc/weight_norm_bwd_cuda.cu',
            'csrc/scale_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O3'] + gencodes})
    ext_modules.append(extension)
else:
    raise RuntimeError("Could not find Cuda install directory")

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
