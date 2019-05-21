import torch
import apex
import argparse
import os
import numpy as np

from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import SyncBatchNorm as sbn

from apex.contrib.groupbn.batch_norm import BatchNorm2d_NHWC as bnp

import time

class base_bn(sbn):
    def __init__(self, planes, fuse_relu=False, bn_group=1):
        super(base_bn, self).__init__(planes, fuse_relu=fuse_relu, channel_last=True, process_group=apex.parallel.create_syncbn_process_group(bn_group))
        torch.nn.init.uniform_(self.weight.data)
        torch.nn.init.uniform_(self.bias.data) # not typical, but setting just to have non zero values for testing
        self.momentum = 1.0

class sync_bn(bnp):
    def __init__(self, planes, fuse_relu=False, bn_group=1):
        super(sync_bn, self).__init__(planes, fuse_relu=fuse_relu, bn_group=bn_group)
        torch.nn.init.uniform_(self.weight.data)
        torch.nn.init.uniform_(self.bias.data) # not typical, but setting just to have non zero values for testing
        self.momentum = 1.0

class block(torch.nn.Module):
    def __init__(self, bn, C, fuse_relu=False, bn_group=1):
        super(block, self).__init__()
        self.bn1 = bn(C, fuse_relu=fuse_relu, bn_group=bn_group)

    def forward(self, x, z = None):
        out = self.bn1(x, z)
        return out

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--bn_group", default=0, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument('--relu', action='store_true')
parser.add_argument('--add', action='store_true')
parser.add_argument("--N", default=16, type=int)
parser.add_argument("--C", default=128, type=int)
parser.add_argument("--HW", default=16, type=int)
args = parser.parse_args()

args.world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

N = args.N
C = args.C
H = args.HW
W = args.HW

if args.add:
  assert(args.relu)

start = (args.local_rank%args.bn_group) * N//args.bn_group
finish = (args.local_rank%args.bn_group + 1) * N//args.bn_group

print("Init batchnorms - setting up IPC channels")
base_block = block(base_bn, C, args.relu, 1).cuda()
sync_block = block(sync_bn, C, args.relu, args.bn_group).cuda()

sync_block = DDP(sync_block, allreduce_always_fp32=True)

base_block.load_state_dict(sync_block.module.state_dict())

base_opt = torch.optim.SGD(base_block.parameters(), lr=1.0)
sync_opt = torch.optim.SGD(sync_block.parameters(), lr=1.0)

print("Generating input data")
np.random.seed(args.seed + int(args.local_rank/args.bn_group))

inp = np.random.normal(1.0, 2.5, (N, H, W, C)).astype(np.float16)
x = torch.cuda.HalfTensor(inp)

# Copy input tensor
x_sync = x[start:finish].clone()

x_sync.requires_grad = True
x.requires_grad = True

if args.add:
    inp = np.random.normal(2.0, 0.5, (N, H, W, C)).astype(np.float16)
    z = torch.cuda.HalfTensor(inp)
    
    # Copy input tensor
    z_sync = z[start:finish].clone()
    
    z_sync.requires_grad = True
    z.requires_grad = True

inp = np.ones((N, H, W, C)).astype(np.float16)*0.02 * 2
g = torch.cuda.HalfTensor(inp)
g_sync = g[start:finish].clone().contiguous()

if args.add:
    base_out = base_block(x, z)
else:
    base_out = base_block(x)
base_out.backward(g)

if args.add:
    sync_out = sync_block(x_sync, z_sync)
else:
    sync_out = sync_block(x_sync)
sync_out.backward(g_sync)


for para in base_block.parameters():
    para.grad = para.grad / args.bn_group

base_opt.step()
sync_opt.step()

# useful routine in case you want to explore differences
def compare(desc, inp1, inp2, error):
    a = inp1.clone().detach().cpu().numpy()
    b = inp2.clone().detach().cpu().numpy()
    close = np.allclose(a,b, error, error)
    if not close:
        index = np.nonzero(np.where(np.isclose(a, b, atol=error, rtol=error), False, True))
        print(desc, close)
        z = a - b
        print(desc, z.shape, len(index[0]));
        base = a[index]
        diff = z[index]
        print(desc, "\n dif    : ", diff, "\n base    : ", base)
    else:
        print(desc, " passed")
    return close

def getIndex(inp1, inp2, error):
    a = inp1.clone().detach().cpu().numpy()
    b = inp2.clone().detach().cpu().numpy()
    return np.nonzero(np.where(np.isclose(a, b, atol=error, rtol=error), False, True))

err = 1e-3

compare("output: ", base_out[start:finish], sync_out, err)
compare("gradient: ", x.grad[start:finish], x_sync.grad, err)

compare("bn1 weight grad: ", sync_block.module.bn1.weight.grad, base_block.bn1.weight.grad, err)
compare("bn1 bias grad: ", sync_block.module.bn1.bias.grad, base_block.bn1.bias.grad, err)

compare("bn1 rm: ", sync_block.module.bn1.running_mean, base_block.bn1.running_mean, err)
compare("bn1 rv: ", sync_block.module.bn1.running_var, base_block.bn1.running_var, err)
compare("bn1 weight: ", sync_block.module.bn1.weight, base_block.bn1.weight, err)
compare("bn1 bias: ", sync_block.module.bn1.bias, base_block.bn1.bias, err)
