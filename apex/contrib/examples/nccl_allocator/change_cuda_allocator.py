import torch
import apex.contrib.nccl_allocator as nccl_allocator

nccl_allocator.init()
nrep = 6
with nccl_allocator.nccl_mem():
    for i in range(nrep):
      out = torch.randn(1024).cuda()

for i in range(nrep):
  out = torch.randn(1024).cuda()

torch.cuda.empty_cache()
torch.cuda.empty_cache()
