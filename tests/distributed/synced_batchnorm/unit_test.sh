python python_single_gpu_unit_test.py || exit 1
python single_gpu_unit_test.py || exit 1
python test_batchnorm1d.py || exit 1
torchrun --nnodes 1 --nproc-per-node 2  two_gpu_unit_test.py || exit 1
torchrun --nnodes 1 --nproc-per-node 2  two_gpu_unit_test.py --fp16  || exit 1
torchrun --nnodes 1 --nproc-per-node 2  two_gpu_test_different_batch_size.py --apex || exit 1
#beware, you need a system with at least 4 gpus to test group_size<world_size
# torchrun --nnodes 1 --nproc-per-node 4 test_groups.py --group_size=2 
