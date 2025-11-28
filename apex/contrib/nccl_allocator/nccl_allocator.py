import os
import torch
import _apex_nccl_allocator

from contextlib import nullcontext


__all__ = ["init", "nccl_mem", "create_nccl_mem_pool"]


def get_func_args(func):
    import inspect

    sig = inspect.signature(func)
    return [arg.name for arg in sig.parameters.values()]


def create_nccl_mem_pool(symmetric: bool | None = None) -> torch.cuda.MemPool:
    _allocator = _apex_nccl_allocator.get_nccl_allocator()
    if symmetric is None:
        _pool = torch.cuda.MemPool(_allocator)
    else:
        if "symmetric" in get_func_args(torch.cuda.MemPool):
            _pool = torch.cuda.MemPool(_allocator, symmetric=symmetric)
        elif "symm_mem" in get_func_args(torch.cuda.MemPool):
            # This path handles argument name divergence between
            # nvidia pytorch and the official pytorch.
            _pool = torch.cuda.MemPool(_allocator, symm_mem=symmetric)
        else:
            raise ValueError(
                "symmetric setting with torch.cuda.MemPool requires higher PyTorch version"
            )
    return _pool


def init() -> None:
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"


class nccl_mem:
    def __init__(self, pool, enabled=True, device=None, group=None):
        self.device = None
        self.group = None
        self.mem_context = None
        self.pool = pool

        if enabled:
            if device is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, int):
                self.device = torch.device("cuda", device)
            elif isinstance(device, str):
                assert "cuda" in device, "only cuda devices are supported"
                self.device = torch.device(device)

            if group is None:
                self.group = torch.distributed.distributed_c10d._get_default_group()
            else:
                self.group = group

            self.mem_context = torch.cuda.use_mem_pool(self.pool)
        else:
            self.mem_context = nullcontext()

    def __enter__(self):
        self.mem_context.__enter__()
        if self.group is not None:
            backend = self.group._get_backend(self.device)
            try:
                backend.deregister_mem_pool(self.pool)
            except RuntimeError:
                pass

    def __exit__(self, *args):
        if self.group is not None:
            backend = self.group._get_backend(self.device)
            try:
                backend.register_mem_pool(self.pool)
            except RuntimeError:
                pass
        self.mem_context.__exit__(*args)
