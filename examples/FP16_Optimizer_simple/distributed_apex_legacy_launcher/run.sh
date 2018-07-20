#!/bin/bash
# By default, apex.parallel.multiproc will attempt to use all available GPUs on the system.  
# The number of GPUs to use can be limited by setting CUDA_VISIBLE_DEVICES:
export CUDA_VISIBLE_DEVICES=0,1
python -m apex.parallel.multiproc distributed_data_parallel.py
