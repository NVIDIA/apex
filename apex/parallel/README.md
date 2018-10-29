## Distributed Data Parallel

distributed.py contains the source code for `apex.parallel.DistributedDataParallel`, a module wrapper that enables multi-process multi-GPU data parallel training optimized for NVIDIA's NCCL communication library.

`apex.parallel.DistributedDataParallel` achieves high performance by overlapping communication with
computation in the backward pass and bucketing smaller transfers to reduce the total number of
transfers required.

multiproc.py contains the source code for `apex.parallel.multiproc`, a launch utility that places one process on each of the node's available GPUs.

#### [API Documentation](https://nvidia.github.io/apex/parallel.html)

#### [Example/Walkthrough](https://github.com/NVIDIA/apex/tree/master/examples/distributed)

#### [Imagenet example with Mixed Precision](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

#### [Simple example with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/FP16_Optimizer_simple/distributed_apex)

### Synchronized Batch Normalization

`apex.parallel.SyncBatchNorm` has similar APIs as with `torch.nn.BatchNorm*N*d`.
It reduces stats on the first (channel) dimension of the Tensor and accepts
arbitrary spatial dimensions.

#### Installation

Apex provides two sync BN implementation:

1. There is the Python-only implementation, which is the default implementation
when install with `python setup.py install`.
It uses PyTorch primitive operations and distributed communication package from
`torch.distributed`.

   - _Python-only implementation requires input tensor to be of same data type as
layer_

2. We also provide implementation with kernels through CUDA/C++ extension with
improved performance. We are experimenting with Welford and Kahan for reduction
hoping to get better accuracy.
   To use the kernel implementation, user need to install Apex with CUDA extension
enabled `python setup.py install --cuda_ext`.

   - _Custom kernel implementation supports fp16 input with fp32 layer as cudnn.
This is required to run imagenet example in fp16._

   - _Currently kernel implementation only supports GPU._

#### HowTo

1. User could use `apex.parallel.SyncBatchNorm` by building their module with
the layer explicitly.

```
import apex
input_t = torch.randn(3, 5, 20).cuda()
sbn = apex.parallel.SyncBatchNorm(5).cuda()
output_t = sbn(input)
```

2. User could also take a constructed `torch.nn.Model` and replace all its `torch.nn.BatchNorm*N*d` modules with `apex.parallel.SyncBatchNorm` through utility function `apex.parallel.convert_syncbn_model`.

```
# model is an instance of torch.nn.Module
import apex
sync_bn_model = apex.parallel.convert_syncbn_model(model)
```
