import torch.cuda
import os
import re
import subprocess
from setuptools import setup, find_packages
from distutils.command.clean import clean
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME

# TODO:  multiple modules, so we don't have to route all interfaces through
# the same interface.cpp file?

if not torch.cuda.is_available():
    print("Warning: Torch did not find available GPUs on this system.\n",
          "If your intention is to cross-compile, this is not an error.")

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("APEx requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

version_le_04 = []
if TORCH_MAJOR == 0 and TORCH_MINOR == 4:
    version_le_04 = ['-DVERSION_LE_04']

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

# Due to https://github.com/pytorch/pytorch/issues/8223, for Pytorch <= 0.4
# torch.utils.cpp_extension's check for CUDA_HOME fails if there are no GPUs
# available on the system, which prevents cross-compiling and building via Dockerfiles.
# Workaround:  manually search for CUDA_HOME if Pytorch <= 0.4.
def find_cuda_home():
    cuda_path = None
    CUDA_HOME = None

    CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        cudart_path = ctypes.util.find_library('cudart')
        if cudart_path is not None:
            cuda_path = os.path.dirname(cudart_path)
        if cuda_path is not None:
            CUDA_HOME = os.path.dirname(cuda_path)
            
    if not cuda_path and not CUDA_HOME:
        nvcc_path = find('/usr/local/', re.compile("nvcc").search, False)
        if nvcc_path:
            CUDA_HOME = os.path.dirname(nvcc_path)
            if CUDA_HOME:
                os.path.dirname(CUDA_HOME)

        if (not os.path.exists(CUDA_HOME+os.sep+"lib64")
            or not os.path.exists(CUDA_HOME+os.sep+"include") ):
            raise RuntimeError("Error: found NVCC at ", 
                               nvcc_path,
                               " but could not locate CUDA libraries"+
                               " or include directories.")
        
        raise RuntimeError("Error: Could not find cuda on this system. " +
                            "Please set your CUDA_HOME enviornment variable "
                            "to the CUDA base directory.")

    return CUDA_HOME

if TORCH_MAJOR == 0 and TORCH_MINOR == 4:
    if CUDA_HOME is None:
        CUDA_HOME = find_cuda_home()
        # Patch cpp_extension's view of CUDA_HOME:
        torch.utils.cpp_extension.CUDA_HOME = CUDA_HOME

def get_cuda_version():
    NVCC = find(CUDA_HOME+os.sep+"bin",
                re.compile('nvcc$').search)
    print("Found NVCC = ", NVCC)

    # Parse output of nvcc to get cuda major version
    nvcc_output = subprocess.check_output([NVCC, '--version']).decode("utf-8")
    CUDA_LIB = re.compile(', V[0-9]+\.[0-9]+\.[0-9]+').search(nvcc_output).group(0).split('V')[1]
    print("Found CUDA_LIB = ", CUDA_LIB)

    CUDA_MAJOR = int(CUDA_LIB.split('.')[0])
    print("Found CUDA_MAJOR = ", CUDA_MAJOR)

    if CUDA_MAJOR < 8:
        raise RuntimeError("APex requires CUDA 8.0 or newer")

    return CUDA_MAJOR

if CUDA_HOME is not None:
 
    print("Found CUDA_HOME = ", CUDA_HOME)

    CUDA_MAJOR = get_cuda_version()

    gencodes = ['-gencode', 'arch=compute_52,code=sm_52',
                '-gencode', 'arch=compute_60,code=sm_60',
                '-gencode', 'arch=compute_61,code=sm_61',]

    if CUDA_MAJOR > 8:
        gencodes += ['-gencode', 'arch=compute_70,code=sm_70',
                     '-gencode', 'arch=compute_70,code=compute_70',]

    ext_modules = []
    extension = CUDAExtension(
        'apex_C', [
            'csrc/interface.cpp',
            'csrc/weight_norm_fwd_cuda.cu',
            'csrc/weight_norm_bwd_cuda.cu',
            'csrc/scale_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g'] + version_le_04,
                            'nvcc': ['-O3'] + version_le_04 + gencodes})
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
