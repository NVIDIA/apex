**distributed_data_parallel.py** and **run.sh** show an example using Amp with
[apex.parallel.DistributedDataParallel](https://nvidia.github.io/apex/parallel.html) or
[torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#distributeddataparallel)
and the Pytorch multiprocess launcher script,
[torch.distributed.launch](https://pytorch.org/docs/master/distributed.html#launch-utility).
The use of `Amp` with DistributedDataParallel does not need to change from ordinary 
single-process use.  The only gotcha is that wrapping your model with `DistributedDataParallel` must
come after the call to `amp.initialize`.  Test via
```bash
bash run.sh
```

**This is intended purely as an instructional example, not a performance showcase.**
