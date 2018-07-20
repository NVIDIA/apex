**distributed_data_parallel.py** and **run.sh** show an example using `FP16_Optimizer` with
`apex.parallel.DistributedDataParallel` in conjuction with the legacy Apex
launcher script, `apex.parallel.multiproc`.  See 
[FP16_Optimizer_simple/distributed_apex](https://github.com/NVIDIA/apex/tree/torch_launcher/examples/FP16_Optimizer_simple/distributed_apex) for a more up-to-date example that uses the Pytorch launcher
script, `torch.distributed.launch`.
The usage of `FP16_Optimizer` with distributed does not need to change from ordinary 
single-process usage.  Test via
```bash
bash run.sh
```
