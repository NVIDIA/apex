import os
import time
import gc
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from torch.nn import BatchNorm2d as GBNREF
from apex.contrib.cudnn_gbn import GroupBatchNorm2d as GBN
import numpy as np

# Usage: torchrun --nproc_per_node 2 tests/distributed/cudnn_gbn/two_gpu_gbn.py

input_shapes = [
    [1, 1024, 48, 72],
    [1, 128, 192, 288],
    [1, 128, 384, 576],
    [1, 1536, 48, 72],
    [1, 2048, 48, 72],
    [1, 256, 1, 1],
    [1, 256, 192, 288],
    [1, 256, 384, 576],
    [1, 256, 48, 72],
    [1, 256, 96, 144],
    [1, 32, 384, 576],
    [1, 48, 192, 288],
    [1, 64, 384, 576],
    [1, 728, 48, 72],
    [1, 728, 96, 144]
 ]

class BNModelRef(nn.Module):
    def __init__(self, num_features, num_layers=1000):
        super().__init__()

        bnlist = [GBNREF(num_features,
                                   eps=1e-05,
                                   momentum=0.1,
                                   affine=True,
                                   track_running_stats=True) for _ in range(num_layers)]

        self.fwd = nn.Sequential(*bnlist)

    def forward(self, x):
        return self.fwd(x)

class BNModel(nn.Module):
    def __init__(self, num_features, num_layers=1000):
        super().__init__()

        bnlist = [GBN(num_features,group_size=2,
                                   eps=1e-05,
                                   momentum=0.1,
                                   affine=True,
                                   track_running_stats=True) for _ in range(num_layers)]

        self.fwd = nn.Sequential(*bnlist)

    def forward(self, x):
        return self.fwd(x)


def get_rand_tensors(global_shape, device):
    inp_t = torch.rand(global_shape, dtype=torch.float32, device=device).to(memory_format=torch.channels_last)
    weight = torch.rand(global_shape[1], dtype=torch.float32, device=device)
    bias = torch.rand(global_shape[1], dtype=torch.float32, device=device)
    _grad_out = torch.rand(global_shape, dtype=torch.float32, device=device).to(memory_format=torch.channels_last)
    return inp_t, weight, bias, _grad_out

def main():
    dist.init_process_group(backend = "nccl")

    # set device
    world_size = dist.get_world_size()
    comm_rank = dist.get_rank()
    comm_local_rank = comm_rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{comm_local_rank}")

    torch.cuda.manual_seed(333)
    torch.manual_seed(333)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    # number of layers
    num_layers = 1

    # create model
    for shape in input_shapes:
        if comm_rank == 0:
            print(f'====Testing shape {shape}====')
        global_shape = shape.copy()
        global_shape[0] = world_size
        model = BNModel(num_features=shape[1],
                        num_layers=num_layers).to(device).to(memory_format=torch.channels_last)
        model_ref = BNModelRef(num_features=shape[1],
                        num_layers=num_layers).to(device).to(memory_format=torch.channels_last)

        inp_t, weight, bias, _grad_out = get_rand_tensors(global_shape, device)

        model_ref.fwd[0].weight.data = weight.clone()
        model_ref.fwd[0].bias.data = bias.clone()
        model.fwd[0].weight.data  = weight.clone()
        model.fwd[0].bias.data  = bias.clone()
        inp_ref = inp_t.clone().requires_grad_()
        inp = inp_t[comm_rank:comm_rank+1,...].clone().requires_grad_()
        grad_out = _grad_out[comm_rank:comm_rank+1,...].half().clone().detach()
        grad_out_ref = _grad_out.half().clone().detach()

        with amp.autocast():
            inp.grad = None
            model.zero_grad()
            out = model(inp)
        out.backward(grad_out)
        torch.distributed.barrier()
        with amp.autocast():
            inp_ref.grad = None
            model_ref.zero_grad()
            out_ref = model_ref(inp_ref.half())
        out_ref.backward(grad_out_ref)

        torch.cuda.current_stream().synchronize()
        rtol = 3.5e-3
        atol = 3e-2
        if comm_rank == 0:
            torch.testing.assert_close(out_ref[comm_rank:comm_rank+1,...], out, rtol=rtol, atol=atol, msg=lambda x: f'Output mismatch\n{x}')
            torch.testing.assert_close(inp_ref.grad[comm_rank:comm_rank+1,...], inp.grad, rtol=rtol, atol=atol, msg=lambda x: f'Input grad mismatch\n{x}')
            # compensating the averaging over processes done by DDP
            # in order to produce mathematically equivalent result
            # https://github.com/NVIDIA/apex/issues/134#issuecomment-458307368
            torch.testing.assert_close(model_ref.fwd[0].bias.grad/ world_size, model.fwd[0].bias.grad, rtol=rtol, atol=atol, msg=lambda x: f'Bias grad mismatch\n{x}')
            torch.testing.assert_close(model_ref.fwd[0].weight.grad/ world_size, model.fwd[0].weight.grad, rtol=rtol, atol=atol, msg=lambda x: f'Weight grad mismatch\n{x}')

if __name__ == "__main__":
    main()