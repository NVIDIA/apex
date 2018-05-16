#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 distributed_data_parallel.py
