import torch
from setuptools import setup, find_packages
import subprocess

import sys

if not torch.cuda.is_available():
    print("\nWarning: Torch did not find available GPUs on this system.\n",
          "If your intention is to cross-compile, this is not an error.\n")

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("Apex requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

cmdclass = {}
ext_modules = []

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
        # TODO:  make this a hard error?
        print("\nWarning:  Cuda extensions are being compiled with a version of Cuda that does "
              "not match the version used to compile Pytorch binaries.\n")
    print("Pytorch binaries were compiled with Cuda {}\n".format(torch.version.cuda))

if "--cuda_ext" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension
    sys.argv.remove("--cuda_ext")

    if torch.utils.cpp_extension.CUDA_HOME is None:
        raise RuntimeError("--cuda_ext was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)

        # Set up macros for forward/backward compatibility hack around
        # https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
        version_ge_1_1 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']

        ext_modules.append(
            CUDAExtension(name='amp_C',
                          sources=['csrc/amp_C_frontend.cpp',
                                   'csrc/multi_tensor_scale_kernel.cu',
                                   'csrc/multi_tensor_axpby_kernel.cu',
                                   'csrc/multi_tensor_l2norm_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3'],
                                              'nvcc':['-lineinfo',
                                                      '-O3',
                                                      # '--resource-usage',
                                                      '--use_fast_math']}))
        ext_modules.append(
            CUDAExtension(name='fused_adam_cuda',
                          sources=['csrc/fused_adam_cuda.cpp',
                                   'csrc/fused_adam_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3',],
                                              'nvcc':['-O3',
                                                      '--use_fast_math']}))
        ext_modules.append(
            CUDAExtension(name='syncbn',
                          sources=['csrc/syncbn.cpp',
                                   'csrc/welford.cu']))
        ext_modules.append(
            CUDAExtension(name='fused_layer_norm_cuda',
                          sources=['csrc/layer_norm_cuda.cpp',
                                   'csrc/layer_norm_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3'] + version_ge_1_1,
                                              'nvcc':['-maxrregcount=50',
                                                      '-O3',
                                                      '--use_fast_math'] + version_ge_1_1}))

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
