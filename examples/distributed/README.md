# Basic Multiprocess Example based on pytorch/examples/mnist

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

[API Documentation](https://nvidia.github.io/apex/parallel.html)

[Source Code](https://github.com/NVIDIA/apex/tree/master/apex/parallel)

[Another Example: Imagenet with mixed precision](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

## Getting started
Prior to running please run
```pip install -r requirements.txt```

To download the dataset, run
```python main.py```
without any arguments.  Once you have downloaded the dataset, you should not need to do this again.

You can now launch multi-process distributed data parallel jobs via
```bash
python -m apex.parallel.multiproc main.py args...
```
adding any `args...` you like.  The launch script `apex.parallel.multiproc` will 
spawn one process for each of your system's available (visible) GPUs.
Each process will run `python main.py args... --world-size <worldsize> --rank <rank>`
(the `--world-size` and `--rank` arguments are determined and appended by `apex.parallel.multiproc`).
Each `main.py` calls `torch.cuda.set_device()` and `torch.distributed.init_process_group()` 
according to the `rank` and `world-size` arguments it receives.

The number of visible GPU devices (and therefore the number of processes 
`DistributedDataParallel` will spawn) can be controlled by setting the environment variable 
`CUDA_VISIBLE_DEVICES`.  For example, if you `export CUDA_VISIBLE_DEVICES=0,1` and run
```python -m apex.parallel.multiproc main.py ...```, the launch utility will spawn two processes
which will run on devices 0 and 1.  By default, if `CUDA_VISIBLE_DEVICES` is unset, 
`apex.parallel.multiproc` will attempt to use every device on the node.

## Converting your own model

To understand how to convert your own model, please see all sections of main.py within ```#=====START: ADDED FOR DISTRIBUTED======``` and ```#=====END:   ADDED FOR DISTRIBUTED======``` flags.

## Requirements
Pytorch master branch built from source. This requirement is to use NCCL as a distributed backend.
