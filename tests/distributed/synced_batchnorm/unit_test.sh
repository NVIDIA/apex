python single_gpu_unit_test.py
python -m torch.distributed.launch --nproc_per_node=2 two_gpu_unit_test.py
python -m torch.distributed.launch --nproc_per_node=2 two_gpu_unit_test.py --fp64
