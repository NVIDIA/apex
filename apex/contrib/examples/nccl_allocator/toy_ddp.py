import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import apex.contrib.nccl_allocator as nccl_allocator

assert os.getenv("WORLD_SIZE") is not None, "Please use: torchrun --nproc-per-node=8 toy_ddp.py"

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


rank = int(os.getenv("RANK"))
local_rank = int(os.getenv("LOCAL_RANK"))
world_size = int(os.getenv("WORLD_SIZE"))

nccl_allocator.init()

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

device = torch.device("cuda", local_rank)
model = ToyModel().to(device)
ddp_model = DDP(model, device_ids=[rank])
loss_fn = nn.MSELoss()
optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

data_ptrs = []
with nccl_allocator.nccl_mem():
    for param in ddp_model.parameters():
        param.grad = torch.empty_like(param)
        data_ptrs.append(param.grad.data_ptr())

for _ in range(10):
    optimizer.zero_grad(set_to_none=False)
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

for data_ptr, param in zip(data_ptrs, ddp_model.parameters()):
    assert(data_ptr == param.grad.data_ptr())
dist.destroy_process_group()
