#!/bin/bash

if [ ! -d "val" ]; then
   ln -sf /data/imagenet/val-jpeg/ val
fi
if [ ! -d "train" ]; then
   ln -sf /data/imagenet/train-jpeg/ train
fi

DATADIR="./"
cp ../common/* .
bash run_test.sh single_gpu $1 $DATADIR yes
