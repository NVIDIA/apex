import torch
from setuptools import setup, find_packages
import subprocess

from pip._internal import main as pipmain
import sys
import warnings
import os

if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n'
          'Volta (compute capability 7.0), and Turing (compute capability 7.5).\n'
          'If you wish to cross-compile for a single specific architecture,\n'
          'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("Apex requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

cmdclass = {}
ext_modules = []

if "--pyprof" in sys.argv:
    with open('requirements.txt') as f:
        required_packages = f.read().splitlines()
        pipmain(["install"] + required_packages)
    try:
        sys.argv.remove("--pyprof")
    except:
        pass
else:
    warnings.warn("Option --pyprof not specified. Not installing PyProf dependencies!")

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    if TORCH_MAJOR == 0:
        raise RuntimeError("--cpp_ext requires Pytorch 1.0 or later, "
                           "found torch.__version__ = {}".format(torch.__version__))
    from torch.utils.cpp_extension import BuildExtension
    cmdclass['build_ext'] = BuildExtension

if "--cpp_ext" in sys.argv:
    from torch.utils.cpp_extension import CppExtension
    sys.argv.remove("--cpp_ext")
    ext_modules.append(
        CppExtension('apex_C',
                     ['csrc/flatten_unflatten.cpp',]))

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does " +
                           "not match the version used to compile Pytorch binaries.  " +
                           "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda) +
                           "In some cases, a minor-version mismatch will not cause later errors:  " +
                           "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
                           "You can try commenting out this check (at your own risk).")

# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_dependent_macros = version_ge_1_1 + version_ge_1_3

if "--cuda_ext" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension
    sys.argv.remove("--cuda_ext")

    if torch.utils.cpp_extension.CUDA_HOME is None:
        raise RuntimeError("--cuda_ext was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)

        ext_modules.append(
            CUDAExtension(name='amp_C',
                          sources=['csrc/amp_C_frontend.cpp',
                                   'csrc/multi_tensor_sgd_kernel.cu',
                                   'csrc/multi_tensor_scale_kernel.cu',
                                   'csrc/multi_tensor_axpby_kernel.cu',
                                   'csrc/multi_tensor_l2norm_kernel.cu',
                                   'csrc/multi_tensor_lamb_stage_1.cu',
                                   'csrc/multi_tensor_lamb_stage_2.cu',
                                   'csrc/multi_tensor_adam.cu',
                                   'csrc/multi_tensor_novograd.cu',
                                   'csrc/multi_tensor_lamb.cu'],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-lineinfo',
                                                      '-O3',
                                                      # '--resource-usage',
                                                      '--use_fast_math'] + version_dependent_macros}))
        ext_modules.append(
            CUDAExtension(name='syncbn',
                          sources=['csrc/syncbn.cpp',
                                   'csrc/welford.cu'],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-O3'] + version_dependent_macros}))

        ext_modules.append(
            CUDAExtension(name='fused_layer_norm_cuda',
                          sources=['csrc/layer_norm_cuda.cpp',
                                   'csrc/layer_norm_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-maxrregcount=50',
                                                      '-O3',
                                                      '--use_fast_math'] + version_dependent_macros}))

if "--bnp" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension
    sys.argv.remove("--bnp")

    from torch.utils.cpp_extension import BuildExtension
    cmdclass['build_ext'] = BuildExtension

    if torch.utils.cpp_extension.CUDA_HOME is None:
        raise RuntimeError("--bnp was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        ext_modules.append(
            CUDAExtension(name='bnp',
                          sources=['apex/contrib/csrc/groupbn/batch_norm.cu',
                                   'apex/contrib/csrc/groupbn/ipc.cu',
                                   'apex/contrib/csrc/groupbn/interface.cpp',
                                   'apex/contrib/csrc/groupbn/batch_norm_add_relu.cu'],
                          include_dirs=['csrc'],
                          extra_compile_args={'cxx': [] + version_dependent_macros,
                                              'nvcc':['-DCUDA_HAS_FP16=1',
                                                      '-D__CUDA_NO_HALF_OPERATORS__',
                                                      '-D__CUDA_NO_HALF_CONVERSIONS__',
                                                      '-D__CUDA_NO_HALF2_OPERATORS__',
                                                      '-gencode',
                                                      'arch=compute_70,code=sm_70'] + version_dependent_macros}))

if "--xentropy" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension
    sys.argv.remove("--xentropy")

    from torch.utils.cpp_extension import BuildExtension
    cmdclass['build_ext'] = BuildExtension

    if torch.utils.cpp_extension.CUDA_HOME is None:
        raise RuntimeError("--xentropy was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        ext_modules.append(
            CUDAExtension(name='xentropy_cuda',
                          sources=['apex/contrib/csrc/xentropy/interface.cpp',
                                   'apex/contrib/csrc/xentropy/xentropy_kernel.cu'],
                          include_dirs=['csrc'],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-O3'] + version_dependent_macros}))

if "--deprecated_fused_adam" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension
    sys.argv.remove("--deprecated_fused_adam")

    from torch.utils.cpp_extension import BuildExtension
    cmdclass['build_ext'] = BuildExtension

    if torch.utils.cpp_extension.CUDA_HOME is None:
        raise RuntimeError("--deprecated_fused_adam was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        ext_modules.append(
            CUDAExtension(name='fused_adam_cuda',
                          sources=['apex/contrib/csrc/optimizers/fused_adam_cuda.cpp',
                                   'apex/contrib/csrc/optimizers/fused_adam_cuda_kernel.cu'],
                          include_dirs=['csrc'],
                          extra_compile_args={'cxx': ['-O3',] + version_dependent_macros,
                                              'nvcc':['-O3',
                                                      '--use_fast_math'] + version_dependent_macros}))

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
