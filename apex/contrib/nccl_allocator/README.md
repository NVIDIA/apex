## General information

`nccl_allocator` is a module that enables `ncclMemAlloc`[^1] to be used within PyTorch for faster NCCL NVLS collective communications.
It is mainly based on `CUDAPluggableAllocator`.
The context manager `nccl_allocator.nccl_mem(enabled=True)` is used as a switch between `cudaMalloc` and `ncclMemAlloc` (if `enabled=True` it will use `cudaMalloc`).

[^1]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html

### Example usage:

Here is a minimalistic example:

```
import os
import torch
import torch.distributed as dist
import apex.contrib.nccl_allocator as nccl_allocator

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
```

Please visit `apex/contrib/examples/nccl_allocator` for more examples.


### IMPORTANT

There are several strict requirements:
- PyTorch must include PR [#112850](https://github.com/pytorch/pytorch/pull/112850)
- NCCL v2.19.4 and newer
- NCCL NVLS requires CUDA Driver 530 and newer (tested on 535)

