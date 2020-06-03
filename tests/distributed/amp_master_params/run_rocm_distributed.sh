#!/bin/bash
set -e

# To run the test on 2 gpus
export WORLD_SIZE=2

# Test with opt_level="O2"
echo "running opt_level O2"
python3.6 -m torch.distributed.launch --nproc_per_node=2 amp_master_params.py --opt_level "O2"
python3.6 compare.py

# delete the model files
echo -e "O2 test completed. Deleting model files\n"
rm rank0model.pth
rm rank1model.pth
rm rank0master.pth
rm rank1master.pth


# Test with opt_level="O5"
echo "running opt_level O5"
python3.6 -m torch.distributed.launch --nproc_per_node=2 amp_master_params.py --opt_level "O5"
python3.6 compare.py

# delete the model files
echo "O5 test completed. Deleting model files"
rm rank0model.pth
rm rank1model.pth
rm rank0master.pth
rm rank1master.pth
