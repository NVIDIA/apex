import sys
import warnings
import os
import glob
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    load,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_cudnn_version_and_warn(global_option: str, required_cudnn_version: int) -> bool:
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    if not (cudnn_available and (cudnn_version >= required_cudnn_version)):
        warnings.warn(
            f"Skip `{global_option}` as it requires cuDNN {required_cudnn_version} or later, "
            f"but {'cuDNN is not available' if not cudnn_available else cudnn_version}"
        )
        return False
    return True


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif bare_metal_version >= Version("11.1"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        elif bare_metal_version == Version("11.0"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
    raise RuntimeError(
        "Apex requires Pytorch 0.4 or newer.\nThe latest stable release can be obtained from https://pytorch.org/"
    )

cmdclass = {}
ext_modules = []

extras = {}

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    if TORCH_MAJOR == 0:
        raise RuntimeError(
            "--cpp_ext requires Pytorch 1.0 or later, " "found torch.__version__ = {}".format(torch.__version__)
        )

if "--cpp_ext" in sys.argv:
    sys.argv.remove("--cpp_ext")
    ext_modules.append(CppExtension("apex_C", ["csrc/flatten_unflatten.cpp"]))


# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ["-DVERSION_GE_1_1"]
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ["-DVERSION_GE_1_3"]
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ["-DVERSION_GE_1_5"]
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)

if "--distributed_adam" in sys.argv:
    sys.argv.remove("--distributed_adam")
    raise_if_cuda_home_none("--distributed_adam")
    ext_modules.append(
        CUDAExtension(
            name="distributed_adam_cuda",
            sources=[
                "apex/contrib/csrc/optimizers/multi_tensor_distopt_adam.cpp",
                "apex/contrib/csrc/optimizers/multi_tensor_distopt_adam_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3", "--use_fast_math"] + version_dependent_macros,
            },
        )
    )

if "--distributed_lamb" in sys.argv:
    sys.argv.remove("--distributed_lamb")
    raise_if_cuda_home_none("--distributed_lamb")
    ext_modules.append(
        CUDAExtension(
            name="distributed_lamb_cuda",
            sources=[
                "apex/contrib/csrc/optimizers/multi_tensor_distopt_lamb.cpp",
                "apex/contrib/csrc/optimizers/multi_tensor_distopt_lamb_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3", "--use_fast_math"] + version_dependent_macros,
            },
        )
    )

if "--cuda_ext" in sys.argv:
    sys.argv.remove("--cuda_ext")
    raise_if_cuda_home_none("--cuda_ext")
    check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

    ext_modules.append(
        CUDAExtension(
            name="amp_C",
            sources=[
                "csrc/amp_C_frontend.cpp",
                "csrc/multi_tensor_sgd_kernel.cu",
                "csrc/multi_tensor_scale_kernel.cu",
                "csrc/multi_tensor_axpby_kernel.cu",
                "csrc/multi_tensor_l2norm_kernel.cu",
                "csrc/multi_tensor_l2norm_kernel_mp.cu",
                "csrc/multi_tensor_l2norm_scale_kernel.cu",
                "csrc/multi_tensor_lamb_stage_1.cu",
                "csrc/multi_tensor_lamb_stage_2.cu",
                "csrc/multi_tensor_adam.cu",
                "csrc/multi_tensor_adagrad.cu",
                "csrc/multi_tensor_novograd.cu",
                "csrc/multi_tensor_lamb.cu",
                "csrc/multi_tensor_lamb_mp.cu",
                "csrc/update_scale_hysteresis.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": [
                    "-lineinfo",
                    "-O3",
                    # '--resource-usage',
                    "--use_fast_math",
                ] + version_dependent_macros,
            },
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="syncbn",
            sources=["csrc/syncbn.cpp", "csrc/welford.cu"],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3"] + version_dependent_macros,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="fused_layer_norm_cuda",
            sources=["csrc/layer_norm_cuda.cpp", "csrc/layer_norm_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-maxrregcount=50", "-O3", "--use_fast_math"] + version_dependent_macros,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="mlp_cuda",
            sources=["csrc/mlp.cpp", "csrc/mlp_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3"] + version_dependent_macros,
            },
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="fused_dense_cuda",
            sources=["csrc/fused_dense.cpp", "csrc/fused_dense_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3"] + version_dependent_macros,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="scaled_upper_triang_masked_softmax_cuda",
            sources=[
                "csrc/megatron/scaled_upper_triang_masked_softmax.cpp",
                "csrc/megatron/scaled_upper_triang_masked_softmax_cuda.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ] + version_dependent_macros,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="generic_scaled_masked_softmax_cuda",
            sources=[
                "csrc/megatron/generic_scaled_masked_softmax.cpp",
                "csrc/megatron/generic_scaled_masked_softmax_cuda.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ] + version_dependent_macros,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="scaled_masked_softmax_cuda",
            sources=["csrc/megatron/scaled_masked_softmax.cpp", "csrc/megatron/scaled_masked_softmax_cuda.cu"],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ] + version_dependent_macros,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="scaled_softmax_cuda",
            sources=["csrc/megatron/scaled_softmax.cpp", "csrc/megatron/scaled_softmax_cuda.cu"],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ] + version_dependent_macros,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="fused_rotary_positional_embedding",
            sources=[
                "csrc/megatron/fused_rotary_positional_embedding.cpp",
                "csrc/megatron/fused_rotary_positional_embedding_cuda.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ] + version_dependent_macros,
            },
        )
    )

    if bare_metal_version >= Version("11.0"):

        cc_flag = []
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_70,code=sm_70")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        if bare_metal_version >= Version("11.1"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_86,code=sm_86")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

        ext_modules.append(
            CUDAExtension(
                name="fused_weight_gradient_mlp_cuda",
                include_dirs=[os.path.join(this_dir, "csrc")],
                sources=[
                    "csrc/megatron/fused_weight_gradient_dense.cpp",
                    "csrc/megatron/fused_weight_gradient_dense_cuda.cu",
                    "csrc/megatron/fused_weight_gradient_dense_16bit_prec_cuda.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"] + version_dependent_macros,
                    "nvcc": [
                        "-O3",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                    ] + version_dependent_macros + cc_flag,
                },
            )
        )

if "--permutation_search" in sys.argv:
    sys.argv.remove("--permutation_search")

    if CUDA_HOME is None:
        raise RuntimeError("--permutation_search was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        cc_flag = ['-Xcompiler', '-fPIC', '-shared']
        ext_modules.append(
            CUDAExtension(name='permutation_search_cuda',
                          sources=['apex/contrib/sparsity/permutation_search_kernels/CUDA_kernels/permutation_search_kernels.cu'],
                          include_dirs=[os.path.join(this_dir, 'apex', 'contrib', 'sparsity', 'permutation_search_kernels', 'CUDA_kernels')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-O3'] + version_dependent_macros + cc_flag}))

if "--bnp" in sys.argv:
    sys.argv.remove("--bnp")
    raise_if_cuda_home_none("--bnp")
    ext_modules.append(
        CUDAExtension(
            name="bnp",
            sources=[
                "apex/contrib/csrc/groupbn/batch_norm.cu",
                "apex/contrib/csrc/groupbn/ipc.cu",
                "apex/contrib/csrc/groupbn/interface.cpp",
                "apex/contrib/csrc/groupbn/batch_norm_add_relu.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": [] + version_dependent_macros,
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ] + version_dependent_macros,
            },
        )
    )

if "--xentropy" in sys.argv:
    from datetime import datetime
    sys.argv.remove("--xentropy")
    raise_if_cuda_home_none("--xentropy")
    xentropy_ver = datetime.today().strftime("%y.%m.%d")
    print(f"`--xentropy` setting version of {xentropy_ver}")
    ext_modules.append(
        CUDAExtension(
            name="xentropy_cuda",
            sources=["apex/contrib/csrc/xentropy/interface.cpp", "apex/contrib/csrc/xentropy/xentropy_kernel.cu"],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros + [f'-DXENTROPY_VER="{xentropy_ver}"'],
                "nvcc": ["-O3"] + version_dependent_macros,
            },
        )
    )

if "--focal_loss" in sys.argv:
    sys.argv.remove("--focal_loss")
    raise_if_cuda_home_none("--focal_loss")
    ext_modules.append(
        CUDAExtension(
            name='focal_loss_cuda',
            sources=[
                'apex/contrib/csrc/focal_loss/focal_loss_cuda.cpp',
                'apex/contrib/csrc/focal_loss/focal_loss_cuda_kernel.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3'] + version_dependent_macros,
                'nvcc':['-O3', '--use_fast_math', '--ftz=false'] + version_dependent_macros,
            },
        )
    )

if "--group_norm" in sys.argv:
    sys.argv.remove("--group_norm")
    raise_if_cuda_home_none("--group_norm")

    # CUDA group norm supports from SM70
    arch_flags = []
    for arch in [70, 75, 80, 86, 90]:
        arch_flag = f"-gencode=arch=compute_{arch},code=sm_{arch}"
        arch_flags.append(arch_flag)
    arch_flag = f"-gencode=arch=compute_90,code=compute_90"
    arch_flags.append(arch_flag)

    ext_modules.append(
        CUDAExtension(
            name="group_norm_cuda",
            sources=[
                "apex/contrib/csrc/group_norm/group_norm_nhwc_op.cpp",
            ] + glob.glob("apex/contrib/csrc/group_norm/*.cu"),
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + version_dependent_macros,
                "nvcc": [
                    "-O3", "-std=c++17", "--use_fast_math", "--ftz=false",
                ] + arch_flags + version_dependent_macros,
            },
        )
    )

if "--index_mul_2d" in sys.argv:
    sys.argv.remove("--index_mul_2d")
    raise_if_cuda_home_none("--index_mul_2d")
    ext_modules.append(
        CUDAExtension(
            name='fused_index_mul_2d',
            sources=[
                'apex/contrib/csrc/index_mul_2d/index_mul_2d_cuda.cpp',
                'apex/contrib/csrc/index_mul_2d/index_mul_2d_cuda_kernel.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3'] + version_dependent_macros,
                'nvcc':['-O3', '--use_fast_math', '--ftz=false'] + version_dependent_macros,
            },
        )
    )

if "--deprecated_fused_adam" in sys.argv:
    sys.argv.remove("--deprecated_fused_adam")
    raise_if_cuda_home_none("--deprecated_fused_adam")
    ext_modules.append(
        CUDAExtension(
            name="fused_adam_cuda",
            sources=[
                "apex/contrib/csrc/optimizers/fused_adam_cuda.cpp",
                "apex/contrib/csrc/optimizers/fused_adam_cuda_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3", "--use_fast_math"] + version_dependent_macros,
            },
        )
    )

if "--deprecated_fused_lamb" in sys.argv:
    sys.argv.remove("--deprecated_fused_lamb")
    raise_if_cuda_home_none("--deprecated_fused_lamb")
    ext_modules.append(
        CUDAExtension(
            name="fused_lamb_cuda",
            sources=[
                "apex/contrib/csrc/optimizers/fused_lamb_cuda.cpp",
                "apex/contrib/csrc/optimizers/fused_lamb_cuda_kernel.cu",
                "csrc/multi_tensor_l2norm_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3", "--use_fast_math"] + version_dependent_macros,
            },
        )
    )

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

if "--fast_layer_norm" in sys.argv:
    sys.argv.remove("--fast_layer_norm")
    raise_if_cuda_home_none("--fast_layer_norm")

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")

    if bare_metal_version >= Version("11.0"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    ext_modules.append(
        CUDAExtension(
            name="fast_layer_norm",
            sources=[
                "apex/contrib/csrc/layer_norm/ln_api.cpp",
                "apex/contrib/csrc/layer_norm/ln_fwd_cuda_kernel.cu",
                "apex/contrib/csrc/layer_norm/ln_bwd_semi_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros + generator_flag,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-I./apex/contrib/csrc/layer_norm/",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ] + version_dependent_macros + generator_flag + cc_flag,
            },
            include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/layer_norm")],
        )
    )

if "--fmha" in sys.argv:
    sys.argv.remove("--fmha")
    raise_if_cuda_home_none("--fmha")

    if bare_metal_version < Version("11.0"):
        raise RuntimeError("--fmha only supported on sm_80 and sm_90 GPUs")

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    ext_modules.append(
        CUDAExtension(
            name="fmhalib",
            sources=[
                "apex/contrib/csrc/fmha/fmha_api.cpp",
                "apex/contrib/csrc/fmha/src/fmha_fill.cu",
                "apex/contrib/csrc/fmha/src/fmha_noloop_reduce.cu",
                "apex/contrib/csrc/fmha/src/fmha_fprop_fp16_128_64_kernel.sm80.cu",
                "apex/contrib/csrc/fmha/src/fmha_fprop_fp16_256_64_kernel.sm80.cu",
                "apex/contrib/csrc/fmha/src/fmha_fprop_fp16_384_64_kernel.sm80.cu",
                "apex/contrib/csrc/fmha/src/fmha_fprop_fp16_512_64_kernel.sm80.cu",
                "apex/contrib/csrc/fmha/src/fmha_dgrad_fp16_128_64_kernel.sm80.cu",
                "apex/contrib/csrc/fmha/src/fmha_dgrad_fp16_256_64_kernel.sm80.cu",
                "apex/contrib/csrc/fmha/src/fmha_dgrad_fp16_384_64_kernel.sm80.cu",
                "apex/contrib/csrc/fmha/src/fmha_dgrad_fp16_512_64_kernel.sm80.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros + generator_flag,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ] + version_dependent_macros + generator_flag + cc_flag,
            },
            include_dirs=[
                os.path.join(this_dir, "apex/contrib/csrc"),
                os.path.join(this_dir, "apex/contrib/csrc/fmha/src"),
            ],
        )
    )


if "--fast_multihead_attn" in sys.argv:
    sys.argv.remove("--fast_multihead_attn")
    raise_if_cuda_home_none("--fast_multihead_attn")

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")

    if bare_metal_version >= Version("11.0"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.1"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_86,code=sm_86")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    subprocess.run(["git", "submodule", "update", "--init", "apex/contrib/csrc/multihead_attn/cutlass"])
    ext_modules.append(
        CUDAExtension(
            name="fast_multihead_attn",
            sources=[
                "apex/contrib/csrc/multihead_attn/multihead_attn_frontend.cpp",
                "apex/contrib/csrc/multihead_attn/additive_masked_softmax_dropout_cuda.cu",
                "apex/contrib/csrc/multihead_attn/masked_softmax_dropout_cuda.cu",
                "apex/contrib/csrc/multihead_attn/encdec_multihead_attn_cuda.cu",
                "apex/contrib/csrc/multihead_attn/encdec_multihead_attn_norm_add_cuda.cu",
                "apex/contrib/csrc/multihead_attn/self_multihead_attn_cuda.cu",
                "apex/contrib/csrc/multihead_attn/self_multihead_attn_bias_additive_mask_cuda.cu",
                "apex/contrib/csrc/multihead_attn/self_multihead_attn_bias_cuda.cu",
                "apex/contrib/csrc/multihead_attn/self_multihead_attn_norm_add_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros + generator_flag,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ]
                + version_dependent_macros
                + generator_flag
                + cc_flag,
            },
            include_dirs=[
                os.path.join(this_dir, "apex/contrib/csrc/multihead_attn/cutlass/include/"),
                os.path.join(this_dir, "apex/contrib/csrc/multihead_attn/cutlass/tools/util/include")
            ],
        )
    )

if "--transducer" in sys.argv:
    sys.argv.remove("--transducer")
    raise_if_cuda_home_none("--transducer")
    ext_modules.append(
        CUDAExtension(
            name="transducer_joint_cuda",
            sources=[
                "apex/contrib/csrc/transducer/transducer_joint.cpp",
                "apex/contrib/csrc/transducer/transducer_joint_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros + generator_flag,
                "nvcc": ["-O3"] + version_dependent_macros + generator_flag,
            },
            include_dirs=[os.path.join(this_dir, "csrc"), os.path.join(this_dir, "apex/contrib/csrc/multihead_attn")],
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="transducer_loss_cuda",
            sources=[
                "apex/contrib/csrc/transducer/transducer_loss.cpp",
                "apex/contrib/csrc/transducer/transducer_loss_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3"] + version_dependent_macros,
            },
        )
    )

if "--cudnn_gbn" in sys.argv:
    sys.argv.remove("--cudnn_gbn")
    raise_if_cuda_home_none("--cudnn_gbn")
    if check_cudnn_version_and_warn("--cudnn_gbn", 8500):
        subprocess.run(["git", "submodule", "update", "--init", "apex/contrib/csrc/cudnn-frontend/"])
        ext_modules.append(
            CUDAExtension(
                name="cudnn_gbn_lib",
                sources=[
                    "apex/contrib/csrc/cudnn_gbn/norm_sample.cpp",
                    "apex/contrib/csrc/cudnn_gbn/cudnn_gbn.cpp",
                ],
                include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/cudnn-frontend/include")],
                extra_compile_args={"cxx": ["-O3", "-g"] + version_dependent_macros + generator_flag},
            )
        )

if "--peer_memory" in sys.argv:
    sys.argv.remove("--peer_memory")
    raise_if_cuda_home_none("--peer_memory")
    ext_modules.append(
        CUDAExtension(
            name="peer_memory_cuda",
            sources=[
                "apex/contrib/csrc/peer_memory/peer_memory_cuda.cu",
                "apex/contrib/csrc/peer_memory/peer_memory.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
        )
    )

# NOTE: Requires NCCL >= 2.10.3
if "--nccl_p2p" in sys.argv:
    sys.argv.remove("--nccl_p2p")
    raise_if_cuda_home_none("--nccl_p2p")
    # Check NCCL version.
    _nccl_version_getter = load(
        name="_nccl_version_getter",
        sources=["apex/contrib/csrc/nccl_p2p/nccl_version.cpp", "apex/contrib/csrc/nccl_p2p/nccl_version_check.cu"],

    )
    _available_nccl_version = _nccl_version_getter.get_nccl_version()
    if _available_nccl_version >= (2, 10):
        ext_modules.append(
            CUDAExtension(
                name="nccl_p2p_cuda",
                sources=[
                    "apex/contrib/csrc/nccl_p2p/nccl_p2p_cuda.cu",
                    "apex/contrib/csrc/nccl_p2p/nccl_p2p.cpp",
                ],
                extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
            )
        )
    else:
        warnings.warn(
            f"Skip `--nccl_p2p` as it requires NCCL 2.10.3 or later, but {_available_nccl_version[0]}.{_available_nccl_version[1]}"
        )

# note (mkozuki): Now `--fast_bottleneck` option (i.e. apex/contrib/bottleneck) depends on `--peer_memory` and `--nccl_p2p`.
if "--fast_bottleneck" in sys.argv:
    sys.argv.remove("--fast_bottleneck")
    raise_if_cuda_home_none("--fast_bottleneck")
    if check_cudnn_version_and_warn("--fast_bottleneck", 8400):
        subprocess.run(["git", "submodule", "update", "--init", "apex/contrib/csrc/cudnn-frontend/"])
        ext_modules.append(
            CUDAExtension(
                name="fast_bottleneck",
                sources=["apex/contrib/csrc/bottleneck/bottleneck.cpp"],
                include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/cudnn-frontend/include")],
                extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
            )
        )


if "--fused_conv_bias_relu" in sys.argv:
    sys.argv.remove("--fused_conv_bias_relu")
    raise_if_cuda_home_none("--fused_conv_bias_relu")
    if check_cudnn_version_and_warn("--fused_conv_bias_relu", 8400):
        subprocess.run(["git", "submodule", "update", "--init", "apex/contrib/csrc/cudnn-frontend/"])
        ext_modules.append(
            CUDAExtension(
                name="fused_conv_bias_relu",
                sources=["apex/contrib/csrc/conv_bias_relu/conv_bias_relu.cpp"],
                include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/cudnn-frontend/include")],
                extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
            )
        )


if "--nccl_allocator" in sys.argv:
    sys.argv.remove("--nccl_allocator")
    raise_if_cuda_home_none("--nccl_allocator")
    _nccl_version_getter = load(
        name="_nccl_version_getter",
        sources=["apex/contrib/csrc/nccl_p2p/nccl_version.cpp", "apex/contrib/csrc/nccl_p2p/nccl_version_check.cu"],
    )
    _available_nccl_version = _nccl_version_getter.get_nccl_version()
    if _available_nccl_version >= (2, 19):
        ext_modules.append(
            CUDAExtension(
                name="nccl_allocator",
                sources=[
                    "apex/contrib/csrc/nccl_allocator/python_bindings.cpp",
                    "apex/contrib/csrc/nccl_allocator/NCCLAllocator.cpp",
                ],
                include_dirs=[os.path.join(this_dir, "apex/apex/contrib/csrc/nccl_allocator")],
                libraries=["nccl"],
                extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
            )
        )
    else:
        warnings.warn(
            f"Skip `--nccl_allocator` as it requires NCCL 2.19 or later, but {_available_nccl_version[0]}.{_available_nccl_version[1]}"
        )


if "--gpu_direct_storage" in sys.argv:
    sys.argv.remove("--gpu_direct_storage")
    raise_if_cuda_home_none("--gpu_direct_storage")
    ext_modules.append(
        CUDAExtension(
            name="_apex_gpu_direct_storage",
            sources=["apex/contrib/csrc/gpu_direct_storage/gds.cpp", "apex/contrib/csrc/gpu_direct_storage/gds_pybind.cpp"],
            include_dirs=[os.path.join(this_dir, "apex/contrib/csrc/gpu_direct_storage")],
            libraries=["cufile"],
            extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + generator_flag},
        )
    )


setup(
    name="apex",
    version="0.1",
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info",)
    ),
    install_requires=["packaging>20.6"],
    description="PyTorch Extensions written by NVIDIA",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    extras_require=extras,
)
