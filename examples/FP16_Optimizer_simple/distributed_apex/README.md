**distributed_data_parallel.py** shows an example using `FP16_Optimizer` with
`apex.parallel.DistributedDataParallel`.
The usage of `FP16_Optimizer` with distributed does not need to change from ordinary 
single-process usage.  Test via
```bash
bash run.sh
```
