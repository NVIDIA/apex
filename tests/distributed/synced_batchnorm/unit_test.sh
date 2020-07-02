python python_single_gpu_unit_test.py
python single_gpu_unit_test.py
python test_batchnorm1d.py
python -m torch.distributed.launch --nproc_per_node=2 two_gpu_unit_test.py
python -m torch.distributed.launch --nproc_per_node=2 two_gpu_unit_test.py --fp16
python -m torch.distributed.launch --nproc_per_node=2 two_gpu_test_different_batch_size.py --apex
#beware, you need a system with at least 4 gpus to test group_size<world_size
#python -m torch.distributed.launch --nproc_per_node=4 test_groups.py --group_size=2
