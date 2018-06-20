distributed.py contains the source code for `apex.parallel.DistributedDataParallel`, a module wrapper that enables multi-process multi-GPU data parallel training optimized for NVIDIA's NCCL communication library.

`apex.parallel.DistributedDataParallel` achieves high performance by overlapping communication with
computation in the backward pass and bucketing smaller transfers to reduce the total number of
transfers required.

multiproc.py contains the source code for `apex.parallel.multiproc`, a launch utility that places one process on each of the node's available GPUs.

#### [API Documentation](https://nvidia.github.io/apex/parallel.html)

#### [Example/Walkthrough](https://github.com/NVIDIA/apex/tree/master/examples/distributed)

#### [Imagenet Example w/Mixed Precision](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)


