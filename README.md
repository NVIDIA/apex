# Introduction

This repository holds NVIDIA-maintained utilities to streamline mixed precision and distributed training in Pytorch.
Some of the code here will be included in upstream Pytorch eventually.
The intent of Apex is to make up-to-date utilities available to users as quickly as possible.

# Installation
Each [`apex.contrib`](./apex/contrib) module requires one or more install options other than `--cpp_ext` and `--cuda_ext`.
Note that contrib modules do not necessarily support stable PyTorch releases, some of them might only be compatible with nightlies.

## Containers
NVIDIA PyTorch Containers are available on NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.
The containers come with all the custom extensions available at the moment. 

See [the NGC documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) for details such as:
- how to pull a container
- how to run a pulled container
- release notes

## From Source

To install Apex from source, we recommend using the nightly Pytorch obtainable from https://github.com/pytorch/pytorch.

The latest stable release obtainable from https://pytorch.org should also work.

We recommend installing [`Ninja`](https://ninja-build.org/) to make compilation faster.

### Linux

For performance and full functionality, we recommend installing Apex with CUDA and C++ extensions using environment variables:

#### Using Environment Variables (Recommended)

```bash
git clone https://github.com/NVIDIA/apex
cd apex
# Build with core extensions (cpp and cuda)
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# To build with additional extensions, specify them with environment variables
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation .

# To build all contrib extensions at once
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_ALL_CONTRIB_EXT=1 pip install -v --no-build-isolation .
```

To reduce the build time, parallel building can be enabled:

```bash
NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
```

When CPU cores or memory are limited, the `--parallel` option is generally preferred over `--threads`. See [pull#1882](https://github.com/NVIDIA/apex/pull/1882) for more details.

#### Using Command-Line Flags (Legacy Method)

The traditional command-line flags are still supported:

```bash
# Using pip config-settings (pip >= 23.1)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# For older pip versions
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# To build with additional extensions
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" ./
```

#### Python-Only Build

APEX also supports a Python-only build via:
```bash
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```
A Python-only build omits:
- Fused kernels required to use `apex.optimizers.FusedAdam`.
- Fused kernels required to use `apex.normalization.FusedLayerNorm` and `apex.normalization.FusedRMSNorm`.
- Fused kernels that improve the performance and numerical stability of `apex.parallel.SyncBatchNorm`.
- Fused kernels that improve the performance of `apex.parallel.DistributedDataParallel` and `apex.amp`.
`DistributedDataParallel`, `amp`, and `SyncBatchNorm` will still be usable, but they may be slower.


### [Experimental] Windows
`pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .` may work if you were able to build Pytorch from source
on your system. A Python-only build via `pip install -v --no-cache-dir .` is more likely to work.  
If you installed Pytorch in a Conda environment, make sure to install Apex in that same environment.


## Custom C++/CUDA Extensions and Install Options

If a requirement of a module is not met, then it will not be built.

|  Module Name  |  Environment Variable  |  Install Option  |  Misc  |
|---------------|------------------------|------------------|--------|
|  `apex_C`     |  `APEX_CPP_EXT=1`      |  `--cpp_ext`     | |
|  `amp_C`      |  `APEX_CUDA_EXT=1`     |  `--cuda_ext`    | |
|  `syncbn`     |  `APEX_CUDA_EXT=1`     |  `--cuda_ext`    | |
|  `fused_layer_norm_cuda`  |  `APEX_CUDA_EXT=1`  |  `--cuda_ext`  | [`apex.normalization`](./apex/normalization) |
|  `mlp_cuda`   |  `APEX_CUDA_EXT=1`     |  `--cuda_ext`    | |
|  `scaled_upper_triang_masked_softmax_cuda`  |  `APEX_CUDA_EXT=1`  |  `--cuda_ext`  | |
|  `generic_scaled_masked_softmax_cuda`  |  `APEX_CUDA_EXT=1`  |  `--cuda_ext`  | |
|  `scaled_masked_softmax_cuda`  |  `APEX_CUDA_EXT=1`  |  `--cuda_ext`  | |
|  `fused_weight_gradient_mlp_cuda`  |  `APEX_CUDA_EXT=1`  |  `--cuda_ext`  | Requires CUDA>=11 |
|  `permutation_search_cuda`  |  `APEX_PERMUTATION_SEARCH=1`  |  `--permutation_search`  | [`apex.contrib.sparsity`](./apex/contrib/sparsity)  |
|  `bnp`        |  `APEX_BNP=1`          |  `--bnp`         |  [`apex.contrib.groupbn`](./apex/contrib/groupbn) |
|  `xentropy`   |  `APEX_XENTROPY=1`     |  `--xentropy`    |  [`apex.contrib.xentropy`](./apex/contrib/xentropy)  |
|  `focal_loss_cuda`  |  `APEX_FOCAL_LOSS=1`  |  `--focal_loss`  |  [`apex.contrib.focal_loss`](./apex/contrib/focal_loss)  |
|  `fused_index_mul_2d`  |  `APEX_INDEX_MUL_2D=1`  |  `--index_mul_2d`  |  [`apex.contrib.index_mul_2d`](./apex/contrib/index_mul_2d)  |
|  `fused_adam_cuda`  |  `APEX_DEPRECATED_FUSED_ADAM=1`  |  `--deprecated_fused_adam`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fused_lamb_cuda`  |  `APEX_DEPRECATED_FUSED_LAMB=1`  |  `--deprecated_fused_lamb`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fast_layer_norm`  |  `APEX_FAST_LAYER_NORM=1`  |  `--fast_layer_norm`  |  [`apex.contrib.layer_norm`](./apex/contrib/layer_norm). different from `fused_layer_norm` |
|  `fmhalib`    |  `APEX_FMHA=1`         |  `--fmha`        |  [`apex.contrib.fmha`](./apex/contrib/fmha)  |
|  `fast_multihead_attn`  |  `APEX_FAST_MULTIHEAD_ATTN=1`  |  `--fast_multihead_attn`  |  [`apex.contrib.multihead_attn`](./apex/contrib/multihead_attn)  |
|  `transducer_joint_cuda`  |  `APEX_TRANSDUCER=1`  |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `transducer_loss_cuda`   |  `APEX_TRANSDUCER=1`  |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `cudnn_gbn_lib`  |  `APEX_CUDNN_GBN=1`  |  `--cudnn_gbn`  | Requires cuDNN>=8.5, [`apex.contrib.cudnn_gbn`](./apex/contrib/cudnn_gbn) |
|  `peer_memory_cuda`  |  `APEX_PEER_MEMORY=1`  |  `--peer_memory`  |  [`apex.contrib.peer_memory`](./apex/contrib/peer_memory)  |
|  `nccl_p2p_cuda`  |  `APEX_NCCL_P2P=1`  |  `--nccl_p2p`  | Requires NCCL >= 2.10, [`apex.contrib.nccl_p2p`](./apex/contrib/nccl_p2p)  |
|  `fast_bottleneck`  |  `APEX_FAST_BOTTLENECK=1`  |  `--fast_bottleneck`  |  Requires `peer_memory_cuda` and `nccl_p2p_cuda`, [`apex.contrib.bottleneck`](./apex/contrib/bottleneck) |
|  `fused_conv_bias_relu`  |  `APEX_FUSED_CONV_BIAS_RELU=1`  |  `--fused_conv_bias_relu`  | Requires cuDNN>=8.4, [`apex.contrib.conv_bias_relu`](./apex/contrib/conv_bias_relu) |
|  `distributed_adam_cuda`  |  `APEX_DISTRIBUTED_ADAM=1`  |  `--distributed_adam`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `distributed_lamb_cuda`  |  `APEX_DISTRIBUTED_LAMB=1`  |  `--distributed_lamb`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `_apex_nccl_allocator`  |  `APEX_NCCL_ALLOCATOR=1`  |  `--nccl_allocator`  | Requires NCCL >= 2.19, [`apex.contrib.nccl_allocator`](./apex/contrib/nccl_allocator)  |
|  `_apex_gpu_direct_storage`  |  `APEX_GPU_DIRECT_STORAGE=1`  |  `--gpu_direct_storage`  |  [`apex.contrib.gpu_direct_storage`](./apex/contrib/gpu_direct_storage)  |

You can also build all contrib extensions at once by setting `APEX_ALL_CONTRIB_EXT=1`.
