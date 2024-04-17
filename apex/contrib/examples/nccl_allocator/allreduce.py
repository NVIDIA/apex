import os
import torch
import torch.distributed as dist
import apex.contrib.nccl_allocator as nccl_allocator

assert os.getenv("WORLD_SIZE") is not None, "Please use: torchrun --nproc-per-node=8 allreduce.py"

rank = int(os.getenv("RANK"))
local_rank = int(os.getenv("LOCAL_RANK"))
world_size = int(os.getenv("WORLD_SIZE"))

nccl_allocator.init()

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

with nccl_allocator.nccl_mem():
    a = torch.ones(1024 * 1024 * 2, device="cuda")
dist.all_reduce(a)

torch.cuda.synchronize()

