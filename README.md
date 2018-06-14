# Introduction

This repository holds NVIDIA-maintained utilities to streamline 
mixed precision and distributed training in Pytorch. 
Some of the code here will be included in upstream Pytorch eventually.
The intention of Apex is to make up-to-date utilities available to 
users as quickly as possible.

# [Full Documentation](https://nvidia.github.io/apex)

# Contents

## 1. Mixed Precision 

[amp:  Automatic Mixed Precision](https://github.com/NVIDIA/apex/tree/master/apex/amp)

`apex.amp` is a tool designed for ease of use and maximum safety in FP16 training.  All potentially unsafe ops are performed in FP32 under the hood, while safe ops are performed using faster, Tensor Core-friendly FP16 math.  `amp` also automatically implements dynamic loss scaling. 

The intention of `amp` is to be the "on-ramp" to easy FP16 training: achieve all the numerical stability of full FP32 training, with most of the performance benefits of full FP16 training.

[FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/apex/fp16_utils)

`apex.FP16_Optimizer` wraps an existing Python optimizer and automatically implements master parameters and static or dynamic loss scaling under the hood.

The intention of `FP16_Optimizer` is to be the "highway" for FP16 training: achieve most of the numerically stability of full FP32 training, and almost all the performance benefits of full FP16 training.

### Examples:

[Simple examples with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/FP16_Optimizer_simple)

[Imagenet with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

[word_language_model with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/word_language_model)

The Imagenet and word_language_model directories also contain examples that show manual management of master parameters and static loss scaling.  
These examples illustrate what sort of operations `amp` and `FP16_Optimizer` are performing automatically.

## 2. Distributed Training

`apex.parallel.DistributedDataParallel` is a module wrapper, similar to 
`torch.nn.parallel.DistributedDataParallel`.  It enables convenient multiprocess distributed training,
optimized for NVIDIA's NCCL communication library.

`apex.parallel.multiproc` is a launch utility that helps set up arguments for `DistributedDataParallel.`

### [Example/Walkthrough](https://github.com/csarofeen/examples/tree/apex/distributed).

### [Python Source](https://nvidia.github.io/apex/parallel).

# Requirements

Python 3

CUDA 9

PyTorch 0.4 or newer.  We recommend to use the latest stable release, obtainable from 
[https://pytorch.org/](https://pytorch.org/).  We also test against the latest master branch, obtainable from [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch).  
If you have any problems building, please file an issue.



# Quick Start

To build the extension run the following command in the root directory of this project
```
python setup.py install
```

To use the extension
```
import apex
```
and optionally (if required for your use)
```
import apex_C as apex_backend
```

<!--
reparametrization and RNN API under construction

Current version of apex contains:
3. Reparameterization function that allows you to recursively apply reparameterization to an entire module (including children modules).
4. An experimental and in development flexible RNN API.
-->


