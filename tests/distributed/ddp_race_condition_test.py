import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.nn import Module
from apex.parallel import DistributedDataParallel as DDP
import argparse


parser = argparse.ArgumentParser(description='allreduce hook example')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='Number of GPUs to use. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')
parser.add_argument('--rank', default=0, type=int,
                    help='Used for multi-process training. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')

args = parser.parse_args()

args.distributed = args.world_size > 1

if args.distributed:
    torch.cuda.set_device(args.rank % torch.cuda.device_count())
    dist.init_process_group(args.dist_backend, 
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

torch.set_printoptions(precision=10)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.a = Parameter(torch.cuda.FloatTensor(4096*4096).fill_(1.0))
        self.b = Parameter(torch.cuda.FloatTensor(4096*4096).fill_(2.0))
    def forward(self, input):
        return (input*self.a)*self.b

model = DDP(Model(), message_size=1)

x = torch.cuda.FloatTensor(4096*4096)

for i in range(10):
    x.fill_(i + args.rank) # fill x with new values every iteration for sanity
    model.zero_grad()
    out = model(x)
    loss = out.sum()
    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("synchronize() + info")
    torch.cuda.synchronize()
    print("i = {}".format(i))
    def info(name, param, val):
        print(name+": grad.data_ptr() = {}, expected sum {}, got {}".format(
              param.grad.data_ptr(), val*4096*4096*(2.*i+1)/2., param.grad.data.sum().item()))
    info("model.a", model.module.a, 2.) 
    info("model.b", model.module.b, 1.)
    torch.cuda.nvtx.range_pop()
