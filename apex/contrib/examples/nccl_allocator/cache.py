import os
import torch
import apex.contrib.nccl_allocator as nccl_allocator
from pynvml.smi import nvidia_smi

def set_device(dev):
    import ctypes
    handle = ctypes.CDLL("libcudart.so")
    result = handle.cudaSetDevice(ctypes.c_int(dev))
    assert result == 0

def print_used_mem(string, nvsmi, device_id = 0):
    print(f"{string}:", nvsmi.DeviceQuery('memory.used')['gpu'][device_id])

nccl_allocator.init()
nrep = 6
nccl_mem = []

set_device(0)
nvsmi = nvidia_smi.getInstance()

print_used_mem("", nvsmi)

with nccl_allocator.nccl_mem():
    for i in range(nrep):
      out = torch.randn(1024 * 1024 * 100).cuda() # >= 400 MB
      nccl_mem.append(out)

print_used_mem("after nccl alloc (+>=2400)", nvsmi) # + 2400+ MB

cudart_mem = []
for i in range(nrep):
  out = torch.randn(1024 * 1024 * 50 ).cuda() # == 200 MB
  cudart_mem.append(out)

print_used_mem("after cudart alloc (+1200)", nvsmi)

del cudart_mem
torch.cuda.empty_cache()
torch.cuda.empty_cache()
print_used_mem("release cudart mem (-1200)", nvsmi) # - 1200 MB

del nccl_mem
nccl_mem2 = []
with nccl_allocator.nccl_mem():
    for i in range(nrep):
      out = torch.randn(1024 * 1024 * 100).cuda() # >= 400 MB
      nccl_mem2.append(out)
print_used_mem("reuse nccl cache (same)", nvsmi) # + 0 MB
del nccl_mem2
torch.cuda.empty_cache()
print_used_mem("release nccl_mem (-2400)", nvsmi) # - 2400 MB

torch.cuda.empty_cache()
