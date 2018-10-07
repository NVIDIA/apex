#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ddp_race_condition_test.py
