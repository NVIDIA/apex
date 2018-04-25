# Introduction

This repo is designed to hold PyTorch modules and utilities that are under active development and experimental. This repo is not designed as a long term solution or a production solution. Things placed in here are intended to be eventually moved to upstream PyTorch.

# Requirements

Python 3
PyTorch 0.3 or newer
CUDA 9

# [Full Documentation](https://nvidia.github.io/apex)

# Quick Start

To build the extension run the following command in the root directory of this project
```
python setup.py install
```

To use the extension simply run
```
import apex
```
and optionally (if required for your use)
```
import apex._C as apex_backend
```

# What's included

Current version of apex contains:
1. Mixed precision utilities can be found [here](https://nvidia.github.io/apex/fp16_utils) examples of using mixed precision utilities can be found for the [PyTorch imagenet example](https://github.com/csarofeen/examples/tree/apex/imagenet) and the [PyTorch word language model example](https://github.com/csarofeen/examples/tree/apex/word_language_model).
2. Parallel utilities can be found [here](https://nvidia.github.io/apex/parallel) and an example/walkthrough can be found [here](https://github.com/csarofeen/examples/tree/apex/distributed)
  - apex/parallel/distributed.py contains a simplified implementation of PyTorch's DistributedDataParallel that's optimized for use with NCCL in single gpu / process mode
  - apex/parallel/multiproc.py is a simple multi-process launcher that can be used on a single node/computer with multiple GPU's
3. Reparameterization function that allows you to recursively apply reparameterization to an entire module (including children modules).
4. An experimental and in development flexible RNN API.



