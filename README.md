# PSA:  Amp 1.0 API coming soon!  
(as introduced by https://info.nvidia.com/webinar-mixed-precision-with-pytorch-reg-page.html.  The `amp` and `FP16_Optimizer` tools currently in master are separate prototypes, which will be unified by the Amp 1.0 API.)

Branch `api_refactor` is tracking my progress.  I will merge to master, along with documentation and examples, by the end of February.

# Introduction

This repository holds NVIDIA-maintained utilities to streamline 
mixed precision and distributed training in Pytorch. 
Some of the code here will be included in upstream Pytorch eventually.
The intention of Apex is to make up-to-date utilities available to 
users as quickly as possible.

## Full API Documentation: [https://nvidia.github.io/apex](https://nvidia.github.io/apex)

# Contents

## 1. Mixed Precision 

### amp:  Automatic Mixed Precision

`apex.amp` is a tool designed for ease of use and maximum safety in FP16 training.  All potentially unsafe ops are performed in FP32 under the hood, while safe ops are performed using faster, Tensor Core-friendly FP16 math.  `amp` also automatically implements dynamic loss scaling. 

The intention of `amp` is to be the "on-ramp" to easy FP16 training: achieve all the numerical stability of full FP32 training, with most of the performance benefits of full FP16 training.

[Python Source and API Documentation](https://github.com/NVIDIA/apex/tree/master/apex/amp)

### FP16_Optimizer

`apex.FP16_Optimizer` wraps an existing Python optimizer and automatically implements master parameters and static or dynamic loss scaling under the hood.

The intention of `FP16_Optimizer` is to be the "highway" for FP16 training: achieve most of the numerically stability of full FP32 training, and almost all the performance benefits of full FP16 training.

[API Documentation](https://nvidia.github.io/apex/fp16_utils.html#automatic-management-of-master-params-loss-scaling)

[Python Source](https://github.com/NVIDIA/apex/tree/master/apex/fp16_utils)

[Simple examples with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/FP16_Optimizer_simple)

[Imagenet with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

[word_language_model with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/word_language_model)

The Imagenet and word_language_model directories also contain examples that show manual management of master parameters and static loss scaling.  

These manual examples illustrate what sort of operations `amp` and `FP16_Optimizer` are performing automatically.

## 2. Distributed Training

`apex.parallel.DistributedDataParallel` is a module wrapper, similar to 
`torch.nn.parallel.DistributedDataParallel`.  It enables convenient multiprocess distributed training,
optimized for NVIDIA's NCCL communication library.

[API Documentation](https://nvidia.github.io/apex/parallel.html)

[Python Source](https://github.com/NVIDIA/apex/tree/master/apex/parallel)

[Example/Walkthrough](https://github.com/NVIDIA/apex/tree/master/examples/distributed)

The [Imagenet with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/imagenet) 
mixed precision examples also demonstrate `apex.parallel.DistributedDataParallel`.

### Synchronized Batch Normalization

`apex.parallel.SyncBatchNorm` extends `torch.nn.modules.batchnorm._BatchNorm` to
support synchronized BN.
It reduces stats across processes during multiprocess distributed data parallel
training.
Synchronous Batch Normalization has been used in cases where only very small
number of mini-batch could be fit on each GPU.
All-reduced stats boost the effective batch size for sync BN layer to be the
total number of mini-batches across all processes.
It has improved the converged accuracy in some of our research models.

# Requirements

Python 3

CUDA 9 or 10

PyTorch 0.4 or newer.  We recommend to use the latest stable release, obtainable from 
[https://pytorch.org/](https://pytorch.org/).  We also test against the latest master branch, obtainable from [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch).  
If you have any problems building, please file an issue.

The cpp and cuda extensions require pytorch 1.0 or newer.



# Quick Start

### Linux
To build the extension run
```
python setup.py install
```
in the root directory of the cloned repository.

To use the extension
```
import apex
```

### CUDA/C++ extension
Apex contains optional CUDA/C++ extensions, installable via
```
python setup.py install [--cuda_ext] [--cpp_ext]
```
Currently, `--cuda_ext` enables
- Fused kernels that improve the performance and numerical stability of `apex.parallel.SyncBatchNorm`.
- Fused kernels required to use `apex.optimizers.FusedAdam`.
- Fused kernels required to use 'apex.normalization.FusedLayerNorm'.

`--cpp_ext` enables
- C++-side flattening and unflattening utilities that reduce the CPU overhead of `apex.parallel.DistributedDataParallel`.

### Windows support
Windows support is experimental, and Linux is recommended.  However, since Apex could be Python-only, there's a good chance the Python-only features "just works" the same way as Linux.  If you installed Pytorch in a Conda environment, make sure to install Apex in that same environment.

<!--
reparametrization and RNN API under construction

Current version of apex contains:
3. Reparameterization function that allows you to recursively apply reparameterization to an entire module (including children modules).
4. An experimental and in development flexible RNN API.
-->
