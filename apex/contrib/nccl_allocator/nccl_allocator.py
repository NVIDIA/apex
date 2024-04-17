import os
import torch
from torch.cuda.memory import _CUDAAllocator
import nccl_allocator

from contextlib import contextmanager

__all__ = ["init", "nccl_mem"]

class NCCLAllocator(_CUDAAllocator):
    def __init__(self):
        self._allocator = nccl_allocator._cuda_create_managed_allocator()

def init():
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "1"
    allocator = NCCLAllocator()
    nccl_allocator._cuda_change_current_allocator(allocator.allocator())

def use_nccl_mem(flag: bool) -> None:
    nccl_allocator._use_nccl_mem(flag)

@contextmanager
def nccl_mem(flag: bool = True) -> None:
    use_nccl_mem(flag)
    yield
    use_nccl_mem(False)
