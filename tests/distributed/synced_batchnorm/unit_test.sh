python python_single_gpu_unit_test.py
python single_gpu_unit_test.py
python -m torch.distributed.launch --nproc_per_node=2 two_gpu_unit_test.py
python -m torch.distributed.launch --nproc_per_node=2 two_gpu_unit_test.py --fp16
#beware, you need a system with at least 4 gpus to test group_size<world_size
#python -m torch.distributed.launch --nproc_per_node=4 test_groups.py --group_size=2
