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
    dist.init_process_group(args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size)
    rank = torch.distributed.get_rank()
torch.set_printoptions(precision=10)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.x = Parameter(torch.cuda.FloatTensor(1,4096*4096).fill_(1.0))
    def forward(self, input):
        return self.x*input
model = DDP(Model(), message_size=1)

z = torch.cuda.FloatTensor(4096*4096)

for i in range(10):
    z.fill_(i + rank) # fill z with new values every iteration for sanity
    model.zero_grad()
    out = model(z)
    loss = out.sum()
    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("synchronize() + sum")
    torch.cuda.synchronize()
    for param in model.parameters():
        print("i = {},\n"
              "param.grad.data_ptr() = {}\n"
              "expected {},\n" 
              "     got {}\n"
              .format(i,
                      param.grad.data_ptr(),
                      4096*4096*(2.*i+1)/2.,
                      param.grad.data.sum().item()))
    torch.cuda.nvtx.range_pop()

