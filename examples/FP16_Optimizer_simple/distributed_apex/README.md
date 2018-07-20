**distributed_data_parallel.py** and **run.sh** show an example using `FP16_Optimizer` with
`apex.parallel.DistributedDataParallel` and the Pytorch multiprocess launcher script,
[torch.distributed.launch](https://pytorch.org/docs/master/distributed.html#launch-utility).
The usage of `FP16_Optimizer` with distributed does not need to change from ordinary 
single-process usage.  Test via
```bash
bash run.sh
```
