# Simple examples of FP16_Optimizer functionality

`minimal.py` shows the basic usage of `FP16_Optimizer`.

`closure.py` shows how to use `FP16_Optimizer` with a closure.

`save_load.py` shows that `FP16_Optimizer` uses the same checkpointing syntax as ordinary Pytorch 
optimizers.

`distributed_pytorch` shows an example using `FP16_Optimizer` with Pytorch DistributedDataParallel.
The usage of `FP16_Optimizer` with distributed does not need to change from ordinary single-process 
usage. Run via
```bash
cd distributed_pytorch
bash run.sh
```

`distributed_pytorch` shows an example using `FP16_Optimizer` with Apex DistributedDataParallel.
Again, the usage of `FP16_Optimizer` with distributed does not need to change from ordinary 
single-process usage.  Run via
```bash
cd distributed_apex
bash run.sh
```
