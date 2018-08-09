python -m torch.distributed.launch --nproc_per_node=4 main.py -a resnet50 --b 64 --workers 4 --sync_batchnorm ./
