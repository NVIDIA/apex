import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import SyncBatchNorm as ApexSyncBatchNorm

import argparse
import os
import numpy as np

var_batch = 16

def compare(desc, inp1, inp2, error= 1e-5):
    a = inp1.clone().detach().cpu().numpy()
    b = inp2.clone().detach().cpu().numpy()
    close = np.allclose(a,b, error, error)
    if not close:
        print(desc, close)
        z = a - b
        index = (np.abs(z) >= error + error * np.abs(b)).nonzero()
        print("dif    : ", z[index])
        print("inp1   : ", a[index])
        print("inp2   : ", b[index])
    return close

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--apex', action='store_true')
args = parser.parse_args()


torch.manual_seed(2809)
# Setup DDP
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda:{}'.format(args.local_rank))

torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    rank=args.local_rank,
)

# Setup model
if args.apex:
    model = nn.Sequential(
        nn.Conv2d(3, 6, 3, 1, 1),
        ApexSyncBatchNorm(6)
    )
else:
    model = nn.Sequential(
        nn.Conv2d(3, 6, 3, 1, 1),
        nn.SyncBatchNorm(6)
    )

# Setup reference model
model_reference = nn.Sequential(
    nn.Conv2d(3, 6, 3, 1, 1),
    nn.BatchNorm2d(6)
)

with torch.no_grad():
    model_reference[0].weight.copy_(model[0].weight)
    model_reference[0].bias.copy_(model[0].bias)
model_reference.to(device)

model = model.to(device)
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

global_batch_size = var_batch + 8
# Create random data
if args.local_rank == 0:
    data = torch.randn(var_batch, 3, 8, 8, device=device, dtype=torch.float) * 50.0
    grad = torch.randint(0, 10, (var_batch, 6, 8, 8), device=device, dtype=torch.float) / 10.0
else:
    data = torch.randn(8, 3, 8, 8, device=device)
    grad = torch.randint(0, 10, (8, 6, 8, 8), device=device, dtype=torch.float) / 10.0

data.requires_grad_()
data.retain_grad = True

weighted_gradient = True 

# DDP forward/backward
output = model(data)

if weighted_gradient:
    output.backward(grad * 2 / global_batch_size)
else:
    output.backward(grad / output.size(0))

d_list = [torch.randn(8, 3, 8, 8, device=device) for i in range(int(os.environ['WORLD_SIZE']))]
y_list = [torch.randn(8, 6, 8, 8, device=device) for i in range(int(os.environ['WORLD_SIZE']))]
dgrad_list = [torch.randn(8, 3, 8, 8, device=device) for i in range(int(os.environ['WORLD_SIZE']))]
grad_list = [torch.randn(8, 6, 8, 8, device=device) for i in range(int(os.environ['WORLD_SIZE']))]
if args.local_rank == 0:
    # placeholder, these random data will later be discarded.
    torch.distributed.all_gather(d_list, torch.randn(8, 3, 8, 8, device=device))
    torch.distributed.all_gather(y_list, torch.randn(8, 6, 8, 8, device=device))
    torch.distributed.all_gather(dgrad_list, torch.randn(8, 3, 8, 8, device=device))
    torch.distributed.all_gather(grad_list, torch.randn(8, 6, 8, 8, device=device))
else:
    torch.distributed.all_gather(d_list, data)
    torch.distributed.all_gather(y_list, output)
    torch.distributed.all_gather(dgrad_list, data.grad)
    torch.distributed.all_gather(grad_list, grad)

torch.distributed.barrier()

if args.local_rank == 0:
    ref_tensor = d_list[1:]
    ref_tensor.insert(0, data)
    assert(ref_tensor[0].equal(data))
    ref_tensor = torch.cat(ref_tensor, 0)
    ref_tensor = ref_tensor.detach()
    ref_tensor.requires_grad_()
    ref_tensor.retain_grad()

    # Reference forward/backward
    output_reference = model_reference(ref_tensor)
    grad_tensor = grad_list[1:]
    grad_tensor.insert(0, grad)
    assert(grad_tensor[0].equal(grad))
    grad_tensor = torch.cat(grad_tensor, 0)
    if weighted_gradient:
        output_reference.backward(grad_tensor / output_reference.size(0))
    else:
        output_reference.backward(grad_tensor / output_reference.size(0))

    dgrad_tensor = dgrad_list[1:]
    dgrad_tensor.insert(0, data.grad)
    dgrad_tensor = torch.cat(dgrad_tensor, 0)
    # check output
    output_tensor = y_list[1:]
    output_tensor.insert(0, output)
    output_tensor = torch.cat(output_tensor, 0)
    passed = True
    passed = passed and compare("check output",
          output_tensor,
          output_reference)
    # check stats
    passed = passed and compare("check running mean failed",
          model_reference[1].running_mean,
          model.module[1].running_mean)
    passed = passed and compare("check running var failed",
          model_reference[1].running_var,
          model.module[1].running_var)
    passed = passed and compare("bn wgrad check failed!",
          model_reference[1].weight.grad,
          model.module[1].weight.grad, 1e-6)
    passed = passed and compare("conv wgrad check failed!",
          model_reference[0].weight.grad,
          model.module[0].weight.grad)
    # can't really compare dgrad directly, as we need to scale it to account for
    # DDP
    # passed = passed and compare("dgrad check failed!", ref_tensor.grad, dgrad_tensor)
    if passed:
      print("====SBN two gpu with different batches test passed")
    else:
      assert("*failed two gpu with different batches tests*")
