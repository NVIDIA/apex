#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m apex.parallel.multiproc ddp_race_condition.py
