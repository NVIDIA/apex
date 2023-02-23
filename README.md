# Introduction

This repository holds NVIDIA-maintained utilities to streamline mixed precision and distributed training in PyTorch.
Some of the code here will be included in upstream PyTorch eventually.
The intent of Apex is to make up-to-date utilities available to users as quickly as possible.

## Full API Documentation: [https://nvidia.github.io/apex](https://nvidia.github.io/apex)

We are going to update the documentation.

# Installation
Each [`apex.contrib`](./apex/contrib) module requires one or more install options other than `--cpp_ext` and `--cuda_ext`.
Note that contrib modules do not necessarily support stable PyTorch releases.

## Containers
NVIDIA PyTorch Containers are available on NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.
The containers come with all the custom extensions available at the moment. 

See [the NGC documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) for details such as:
- how to pull a container
- how to run a pulled container
- release notes

## From Source

To install apex from source, we recommend using the nightly PyTorch obtainable from https://github.com/pytorch/pytorch.

The latest stable release obtainable from https://pytorch.org should also work.

### Linux
For performance and full functionality, we recommend installing Apex with
CUDA and C++ extensions via
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

APEX also supports a Python-only build via
```bash
pip install -v --disable-pip-version-check --no-cache-dir ./
```
A Python-only build omits:
- Fused kernels required to use `apex.optimizers.FusedAdam`.
- Fused kernels required to use `apex.normalization.FusedLayerNorm` and `apex.normalization.FusedRMSNorm`.
- Fused kernels that improve the performance and numerical stability of `apex.parallel.SyncBatchNorm`.
- Fused kernels that improve the performance of `apex.parallel.DistributedDataParallel` and `apex.amp`.
`DistributedDataParallel`, `amp`, and `SyncBatchNorm` will still be usable, but they may be slower.


### [Experimental] Windows
`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .` may work if you were able to build PyTorch from source
on your system. A Python-only build via `pip install -v --no-cache-dir .` is more likely to work.  
If you installed PyTorch in a Conda environment, make sure to install Apex in that same environment.


## Custom C++/CUDA Extensions and Install Options

If a requirement of a module is not met, then it will not be built.

|  Module Name  |  Install Option  |  Misc  |
|---------------|------------------|--------|
|  `apex_C`     |  `--cpp_ext`     | |
|  `amp_C`      |  `--cuda_ext`    | |
|  `syncbn`     |  `--cuda_ext`    | |
|  `fused_layer_norm_cuda`  |  `--cuda_ext`  | [`apex.normalization`](./apex/normalization) |
|  `mlp_cuda`   |  `--cuda_ext`    | |
|  `scaled_upper_triang_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `generic_scaled_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `scaled_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `fused_weight_gradient_mlp_cuda`  |  `--cuda_ext`  | Requires CUDA>=11 |
|  `permutation_search_cuda`  |  `--permutation_search`  | [`apex.contrib.sparsity`](./apex/contrib/sparsity)  |
|  `bnp`        |  `--bnp`         |  [`apex.contrib.groupbn`](./apex/contrib/groupbn) |
|  `xentropy`   |  `--xentropy`    |  [`apex.contrib.xentropy`](./apex/contrib/xentropy)  |
|  `focal_loss_cuda`  |  `--focal_loss`  |  [`apex.contrib.focal_loss`](./apex/contrib/focal_loss)  |
|  `fused_index_mul_2d`  |  `--index_mul_2d`  |  [`apex.contrib.index_mul_2d`](./apex/contrib/index_mul_2d)  |
|  `fused_adam_cuda`  |  `--deprecated_fused_adam`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fused_lamb_cuda`  |  `--deprecated_fused_lamb`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fast_layer_norm`  |  `--fast_layer_norm`  |  [`apex.contrib.layer_norm`](./apex/contrib/layer_norm). different from `fused_layer_norm` |
|  `fmhalib`    |  `--fmha`        |  [`apex.contrib.fmha`](./apex/contrib/fmha)  |
|  `fast_multihead_attn`  |  `--fast_multihead_attn`  |  [`apex.contrib.multihead_attn`](./apex/contrib/multihead_attn)  |
|  `transducer_joint_cuda`  |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `transducer_loss_cuda`   |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `cudnn_gbn_lib`  |  `--cudnn_gbn`  | Requires cuDNN>=8.5, [`apex.contrib.cudnn_gbn`](./apex/contrib/cudnn_gbn) |
|  `peer_memory_cuda`  |  `--peer_memory`  |  [`apex.contrib.peer_memory`](./apex/contrib/peer_memory)  |
|  `nccl_p2p_cuda`  |  `--nccl_p2p`  | Requires NCCL >= 2.10, [`apex.contrib.nccl_p2p`](./apex/contrib/nccl_p2p)  |
|  `fast_bottleneck`  |  `--fast_bottleneck`  |  Requires `peer_memory_cuda` and `nccl_p2p_cuda`, [`apex.contrib.bottleneck`](./apex/contrib/bottleneck) |
|  `fused_conv_bias_relu`  |  `--fused_conv_bias_relu`  | Requires cuDNN>=8.4, [`apex.contrib.conv_bias_relu`](./apex/contrib/conv_bias_relu) |
