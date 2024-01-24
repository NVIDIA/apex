#!/bin/bash -x
set -e

# To run the test on 2 gpus
export WORLD_SIZE=2

torchrun=`dirname \`which python\``/torchrun

# Test with opt_level="O2"
echo "running opt_level O2"
# python -m torch.distributed.launch --nproc_per_node=2 amp_master_params/amp_master_params.py --opt_level "O2"
python $torchrun --nproc_per_node=2 amp_master_params/amp_master_params.py --opt_level "O2"
python amp_master_params/compare.py

# delete the model files
echo -e "O2 test completed. Deleting model files\n"
rm rank0model.pth
rm rank1model.pth
rm rank0master.pth
rm rank1master.pth


# Test with opt_level="O5"
#echo "running opt_level O5"
#python -m torch.distributed.launch --nproc_per_node=2 amp_master_params/amp_master_params.py --opt_level "O5"
#python amp_master_params/compare.py

## delete the model files
#echo "O5 test completed. Deleting model files"
#rm rank0model.pth
#rm rank1model.pth
#rm rank0master.pth
#rm rank1master.pth

## Run the Sync BN Tests.
echo "Running syncbn tests"
python -m torch.distributed.launch --nproc_per_node=2 synced_batchnorm/two_gpu_unit_test.py
python -m torch.distributed.launch --nproc_per_node=2 synced_batchnorm/two_gpu_unit_test.py --fp16
python -m torch.distributed.launch --nproc_per_node=2 synced_batchnorm/two_gpu_test_different_batch_size.py --apex
echo "Running syncbn python only tests"
python synced_batchnorm/python_single_gpu_unit_test.py
echo "Running syncbn batchnorm1d tests"
python synced_batchnorm/test_batchnorm1d.py 
#beware, you need a system with at least 4 gpus to test group_size<world_size    (currently fail both on upstream and rocm fork)
#python -m torch.distributed.launch --nproc_per_node=4 test_groups.py --group_size=2

## Run the DDP Tests
echo "running DDP tests"
HIP_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 DDP/ddp_race_condition_test.py
