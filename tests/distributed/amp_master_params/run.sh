#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 amp_master_params.py

python compare.py
