# Multiprocess Example based on pytorch/examples/mnist

main.py demonstrates how to modify a simple model to enable multiprocess distributed data parallel
training using the module wrapper `apex.parallel.DistributedDataParallel`
(similar to `torch.nn.parallel.DistributedDataParallel`).

Multiprocess distributed data parallel training frequently outperforms single-process 
data parallel training (such as that offered by `torch.nn.DataParallel`) because each process has its 
own python interpreter.  Therefore, driving multiple GPUs with multiple processes reduces 
global interpreter lock contention versus having a single process (with a single GIL) drive all GPUs.

`apex.parallel.DistributedDataParallel` is optimized for use with NCCL.  It achieves high performance by 
overlapping communication with computation during ``backward()`` and bucketing smaller gradient
transfers to reduce the total number of transfers required.

#### [API Documentation](https://nvidia.github.io/apex/parallel.html)

#### [Source Code](https://github.com/NVIDIA/apex/tree/master/apex/parallel)

#### [Another example: Imagenet with mixed precision](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

#### [Simple example with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/FP16_Optimizer_simple/distributed_apex)

## Getting started
Prior to running please run
```pip install -r requirements.txt```

To download the dataset, run
```python main.py```
without any arguments.  Once you have downloaded the dataset, you should not need to do this again.

`main.py` runs multiprocess distributed data parallel jobs using the Pytorch launcher script
[torch.distributed.launch](https://pytorch.org/docs/master/distributed.html#launch-utility).
Jobs are launched via
```bash
python -m torch.distributed.launch --nproc_per_node=N main.py args...
```
`torch.distributed.launch` spawns `N` processes, each of which runs as
`python main.py args... --local_rank <rank>`.
The `local_rank` argument for each process is determined and appended by `torch.distributed.launch`,
and varies between  0 and `N-1`.  `torch.distributed.launch` also provides environment variables 
for each process.
Internally, each process calls `set_device` according to its local
rank and `init_process_group` with `init_method=`env://' to ingest the provided environment 
variables.
For best performance, set `N` equal to the number of visible CUDA devices on the node.

## Converting your own model

To understand how to convert your own model, please see all sections of main.py within ```#=====START: ADDED FOR DISTRIBUTED======``` and ```#=====END:   ADDED FOR DISTRIBUTED======``` flags.

## Requirements
Pytorch with NCCL available as a distributed backend.  Pytorch 0.4+, installed as a pip or conda package, should have this by default.  Otherwise, you can build Pytorch from source, in an environment where NCCL is installed and visible.
