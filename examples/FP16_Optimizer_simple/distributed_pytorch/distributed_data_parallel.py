import torch
from torch.autograd import Variable
import argparse
from apex.fp16_utils import FP16_Optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

x = Variable(torch.cuda.FloatTensor(N, D_in ).normal_()).half()
y = Variable(torch.cuda.FloatTensor(N, D_out).normal_()).half()

model = torch.nn.Linear(D_in, D_out).cuda().half()
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
### CONSTRUCT FP16_Optimizer ###
optimizer = FP16_Optimizer(optimizer)
###

loss_fn = torch.nn.MSELoss()

for t in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    ### CHANGE loss.backward() TO: ###
    optimizer.backward(loss)
    ###
    optimizer.step()

print("final loss = ", loss)

