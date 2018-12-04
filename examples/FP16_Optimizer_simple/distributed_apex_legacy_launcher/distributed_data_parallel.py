import torch
import argparse
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import FP16_Optimizer

parser = argparse.ArgumentParser()
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--world-size', default=2, type=int,
                    help='Number of distributed processes.')
parser.add_argument("--rank", type=int,
                    help='Rank of this process')

args = parser.parse_args()

torch.cuda.set_device(args.rank)
torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.rank)

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

x = torch.randn(N, D_in, device='cuda', dtype=torch.half)
y = torch.randn(N, D_out, device='cuda', dtype=torch.half)

model = torch.nn.Linear(D_in, D_out).cuda().half()
model = DDP(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
### Construct FP16_Optimizer ###
optimizer = FP16_Optimizer(optimizer)
###

loss_fn = torch.nn.MSELoss()

for t in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred.float(), y.float())
    ### Change loss.backward() to: ###
    optimizer.backward(loss)
    ###
    optimizer.step()

print("final loss = ", loss)

