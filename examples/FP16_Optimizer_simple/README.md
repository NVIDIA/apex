# Simple examples of FP16_Optimizer functionality

`minimal.py` shows the basic usage of `FP16_Optimizer` with either static or dynamic loss scaling.  Test via
```bash
python minimal.py
```

`FP16_Optimizer` supports closures with the same control flow as ordinary Pytorch optimizers.  
`closure.py` shows an example.  Test via
```bash
python closure.py
```
See [the API documentation](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.FP16_Optimizer.step) for more details.

`FP16_Optimizer` also supports checkpointing with the same control flow as ordinary Pytorch optimizers.
`save_load.py` shows an example.  Test via
```bash
python save_load.py
```
See [the API documentation](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.FP16_Optimizer.load_state_dict) for more details.

**distributed_pytorch** shows an example using `FP16_Optimizer` with Pytorch DistributedDataParallel.
The usage of `FP16_Optimizer` with distributed does not need to change from ordinary single-process 
usage. Run via
```bash
cd distributed_pytorch
bash run.sh
```

**distributed_pytorch** shows an example using `FP16_Optimizer` with Apex DistributedDataParallel.
Again, the usage of `FP16_Optimizer` with distributed does not need to change from ordinary 
single-process usage.  Run via
```bash
cd distributed_apex
bash run.sh
```
