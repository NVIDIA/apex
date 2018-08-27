import torch
import torch.nn as nn
import argparse

from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import SyncBatchNorm
import numpy as np
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

torch.manual_seed(1)

num_feature = 2
num_batch = 4
data_dim = (num_batch, num_feature, 2, 1, 1)

bn_layer = None
bn_w = None
bn_b = None

# broadcasting parameters
bn_layer = nn.BatchNorm3d(num_feature).cuda()
bn_w = bn_layer.weight
bn_b = bn_layer.bias


# creating DDP sync_batchnorm_layer
sbn_layer = SyncBatchNorm(num_feature).cuda()

sbn_layer.weight.data = bn_w.clone()
sbn_layer.bias.data = bn_b.clone()

sbn_opt = optim.SGD(sbn_layer.parameters(), lr=0.01)

sbn_layer = DDP(sbn_layer)

# prepare input data for each node
data = torch.randn(*data_dim)*255.0

input1 = data.clone().cuda()
input1.requires_grad_(True)

grad = torch.randn(*data_dim)*255.0
output_grad = grad.cuda()
start = args.local_rank * num_batch//2
finish = (args.local_rank + 1) * num_batch//2
out = sbn_layer(input1[start:finish])
out.backward(output_grad[start:finish])

sbn_opt.step()

bn_opt = optim.SGD(bn_layer.parameters(), lr=0.01)
input2 = data.clone().cuda().requires_grad_(True)

# feed input
out2 = bn_layer(input2)

# concat output gradient and feed it to the back path
out2.backward(grad.clone().cuda())
bn_opt.step()
    

def compare(desc, inp1, inp2):
    close = np.allclose(inp1.detach().cpu().numpy(),
                        inp2.detach().cpu().numpy(),
                        1e-4, 1e-4)
    if not close:
        print(inp1, inp2, "not close")
    print(desc, close)
    return close

result = True
print ("-----sanity check----")

# prepare input data for inference
#for flag1, flag2 in list(itertools.product([True, False], repeat=2)):
for flag2 in [True, False]:
    with torch.no_grad():
        data2 = torch.randn(*data_dim)*255.0
        input_inference = data2.clone().cuda()
        start = args.local_rank * num_batch//2
        finish = (args.local_rank + 1) * num_batch//2
        sbn_layer.module.training = False
        sbn_layer.module.track_running_stats = flag2
        out_inference = sbn_layer(input_inference[start:finish])
        
        input2_inference = data2.clone().cuda()
        bn_layer.training = False
        bn_layer.track_running_stats = flag2 
        out2_inference = bn_layer(input2_inference)
    
    # comparing inference output
    result = (result and
              compare("compare inference output equal: ",
                      out_inference, out2_inference[start:finish]
              )
    )

# comparing output
result = (result and
          compare("compare output equal: ",
                  out, out2[start:finish]
          )
)

# comparing input gradient (concat for input gradient)
result = (result and
          compare("compare input gradient equal: ",
                  input1.grad[start:finish],
                  input2.grad[start:finish]
          )
)


if args.local_rank == 0:

    # comparing running_var
    result = (result and
              compare("compare running_var equal: ",
                      bn_layer.running_var,
                      sbn_layer.module.running_var
              )
    )
    
    # comparing running_mean
    result = (result and
              compare("compare running_mean equal: ",
                      bn_layer.running_mean,
                      sbn_layer.module.running_mean
              )
    )

    # comparing bias gradient (mean for bias gradient)
    result = (result and
              compare("compare layer parameter bias equal: ",
                      bn_layer.bias.grad,
                      sbn_layer.module.bias.grad
              )
    )

    # comparing weight gradient (mean for weight gradient)
    result = (result and
              compare("compare layer parameter weight equal: ",
                      bn_layer.weight.grad,
                      sbn_layer.module.weight.grad
              )
    )

    # comparing updated weight
    result = (result and
              compare("compare layer weight equal: ",
                      bn_layer.weight,
                      sbn_layer.module.weight
              )
    )

    # comparing updated bias
    result = (result and
              compare("compare layer bias equal: ",
                      bn_layer.bias,
                      sbn_layer.module.bias
              )
    )
    
if result:
    print("passed all test!")
else:
    raise RuntimeError("test failed")
