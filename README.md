# Introduction

This repository holds ROCm variant of Nvidia's Apex: https://github.com/NVIDIA/apex. 
The aim of Apex repository is to streamline mixed precision and distributed training in Pytorch.
Some of the code here will be included in upstream Pytorch eventually.
The intent of Apex is to make up-to-date utilities available to users as quickly as possible.

## Full API Documentation: [https://nvidia.github.io/apex](https://nvidia.github.io/apex)

## [GTC 2019](https://github.com/mcarilli/mixed_precision_references/tree/master/GTC_2019) and [Pytorch DevCon 2019](https://github.com/mcarilli/mixed_precision_references/tree/master/Pytorch_Devcon_2019) Slides

# Contents

## 1. Amp:  Automatic Mixed Precision

`apex.amp` is a tool to enable mixed precision training by changing only 3 lines of your script.
Users can easily experiment with different pure and mixed precision training modes by supplying
different flags to `amp.initialize`.

[Webinar introducing Amp](https://info.nvidia.com/webinar-mixed-precision-with-pytorch-reg-page.html)
(The flag `cast_batchnorm` has been renamed to `keep_batchnorm_fp32`).

[API Documentation](https://nvidia.github.io/apex/amp.html)

[Comprehensive Imagenet example](https://github.com/rocm/apex/tree/master/examples/imagenet)

[DCGAN example coming soon...](https://github.com/rocm/apex/tree/master/examples/dcgan)

[Moving to the new Amp API](https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users) (for users of the deprecated "Amp" and "FP16_Optimizer" APIs)

## 2. Distributed Training

`apex.parallel.DistributedDataParallel` is a module wrapper, similar to
`torch.nn.parallel.DistributedDataParallel`.  It enables convenient multiprocess distributed training,
optimized for NVIDIA's NCCL communication library.

[API Documentation](https://nvidia.github.io/apex/parallel.html)

[Python Source](https://github.com/rocm/apex/tree/master/apex/parallel)

[Example/Walkthrough](https://github.com/rocm/apex/tree/master/examples/simple/distributed)

The [Imagenet example](https://github.com/rocm/apex/tree/master/examples/imagenet)
shows use of `apex.parallel.DistributedDataParallel` along with `apex.amp`.

### Synchronized Batch Normalization

`apex.parallel.SyncBatchNorm` extends `torch.nn.modules.batchnorm._BatchNorm` to
support synchronized BN.
It allreduces stats across processes during multiprocess (DistributedDataParallel) training.
Synchronous BN has been used in cases where only a small
local minibatch can fit on each GPU.
Allreduced stats increase the effective batch size for the BN layer to the
global batch size across all processes (which, technically, is the correct
formulation).
Synchronous BN has been observed to improve converged accuracy in some of our research models.

### Checkpointing

To properly save and load your `amp` training, we introduce the `amp.state_dict()`, which contains all `loss_scalers` and their corresponding unskipped steps,
as well as `amp.load_state_dict()` to restore these attributes.

In order to get bitwise accuracy, we recommend the following workflow:
```python
# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
torch.save(checkpoint, 'amp_checkpoint.pt')
...

# Restore
model = ...
optimizer = ...
checkpoint = torch.load('amp_checkpoint.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])

# Continue training
...
```

Note that we recommend restoring the model using the same `opt_level`. Also note that we recommend calling the `load_state_dict` methods after `amp.initialize`.

# Installation

## Containers
ROCm pytorch containers are available from https://hub.docker.com/r/rocm/pytorch.

## From Source

To install Apex from source, we recommend using the nightly Pytorch obtainable from https://github.com/rocm/pytorch.

The latest stable release obtainable from https://pytorch.org should also work.

## ROCm
Apex on ROCm supports both python only build and extension build.
Note: Pytorch version recommended is >=1.5 for extension build.

### To install using python only build use the following command in apex folder:
```
python setup.py install
```

=======
### Supported Versions
| ``APEX Version`` | ``APEX branch`` | ``Torch Version`` |
|------------------|-----------------|-------------------|
| ``1.9.0``        | release/1.9.0   | ``2.9``           | 
| ``1.8.0``        | release/1.8.0   | ``2.8``           | 
| ``1.7.0``        | release/1.7.0   | ``2.7``           | 
| ``1.6.0``        | release/1.6.0   | ``2.6``           | 
| ``1.5.0``        | release/1.5.0   | ``2.5``           | 
| ``1.4.0``        | release/1.4.0   | ``2.4``           | 
| ``1.3.0``        | release/1.3.0   | ``2.3``           | 
| ``1.2.0``        | release/1.2.0   | ``2.2``           | 
| ``1.1.0``        | release/1.1.0   | ``2.1``           |
| ``1.0.0``        | release/1.0.0   | ``2.0`` and older |


The relation between APEX and ROCm PyTorch is maintained in file `related_commits` in [ROCm PyTorch release branches](https://github.com/ROCm/pytorch/branches/all?query=release) in the following format. 

```
ubuntu|pytorch|apex|release/1.0.0|06c33eee43f7a22f3ed7d9c3e5be0ddd757dc345|https://github.com/ROCmSoftwarePlatform/apex
centos|pytorch|apex|release/1.0.0|06c33eee43f7a22f3ed7d9c3e5be0ddd757dc345|https://github.com/ROCmSoftwarePlatform/apex
```

### To install using extensions enabled use the following command in apex folder:
```
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
python setup.py install --cpp_ext --cuda_ext

```
Note that using --cuda_ext flag to install Apex will also enable all the extensions supported on ROCm including "--distributed_adam", "--distributed_lamb", "--bnp", "--xentropy", "--deprecated_fused_adam", "--deprecated_fused_lamb", and "--fast_multihead_attn".

In addition, aiter backend can be built during apex installation by providing --aiter flag
```
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--aiter" ./
# otherwise
python setup.py install --cpp_ext --cuda_ext --aiter
```

To use aiter in fused rope, you can use the flag ```USE_ROCM_AITER_ROPE_BACKEND=1```.

### Enable hipblasLT on ROCm
hipblasLT is supported only on mi300 (gfx942) only.  
python setup.py automatically builds apex with hipblasLT support only if GPU device id is gfx942  
To verify if hipblasLT support is enabled, check the build logs  
INFO: IS_HIPBLASLT_SUPPORTED value is True  ==> indicates apex is built with hipblasLT support  
INFO: IS_HIPBLASLT_SUPPORTED value is False  

### Linux
For performance and full functionality, we recommend installing Apex with
CUDA and C++ extensions via
```bash
git clone https://github.com/rocm/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Apex also supports a Python-only build via
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
`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .` may work if you were able to build Pytorch from source
on your system. A Python-only build via `pip install -v --no-cache-dir .` is more likely to work.  
If you installed Pytorch in a Conda environment, make sure to install Apex in that same environment.


# Release notes

# Release notes
## release/1.9.0

- No new features were added in this release cycle.

## release/1.8.0

Unit test related
- Fix transformer unit tests
- Fix fused dense gelu dense unit tests

## release/1.7.0

Build and installation related
- Support use of BUILD_VERSION environment to override version.txt when creating apex wheels
- Disable aiter installation by default. make aiter command is used to build apex

Unit test related
- Include running transformer tests in L0/run_test.py
- Fix transformer unit tests
- Fix batch norm unit tests
- Fix fused dense gelu dense unit tests

## release/1.6.0

Upgraded extensions
- Support unscale_grads in transformer Grad scaler
- Support amp function in fused dense, mlp
- Support blas backend flag in fused dense 
- Support not destroying process group for distributed tests
- Upgrade fused adam to support parameters - capturable, master weights, grad scaler
- Upgrade distributed fused adam to support bias_correction, adam_w_mode, overlap_param_sync, store_params, store_param_remainders, with_scaled_states, nccl_ub
- Upgrade distributed fused lamb to support parameters fused_norm, full_ar, set_param_views_to_flat_buffer, skip_allgather, fuse_scale, param_order, nccl_allgather_channels

Unit test related
- Fix fused dense, fused rope, mlp unit tests
- Add test fused adam unit test
- Include running fused dense tests in L0/run_test.py


## release/1.5.0

Added extensions
- fused bias swiglu
- fused gradient accumulator
- fused rope
  
Upgraded extensions
- Support blaslt backend in fused weight gradient dense module



