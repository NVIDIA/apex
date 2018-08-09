import torch
import torch.nn as nn
import argparse

from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import SyncBatchNorm
import numpy as np
import torch.optim as optim


def printgrad(self, grad_input, grad_output):
    if torch.distributed.get_rank() == 0:
        print("inside + " + self.__class__.__name__ + ' backward')
        print('grad input: ', grad_input)
        print('grad output: ', grad_output)


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

num_feature = 2
num_batch = 2
data_dim = [num_batch, num_feature, 2, 1, 1]

bn_layer = None
bn_w = None
bn_b = None

# broadcasting parameters
if args.local_rank == 0:
    bn_layer = nn.BatchNorm3d(num_feature).cuda()
    bn_w = bn_layer.weight
    bn_b = bn_layer.bias
else:
    bn_w = torch.nn.parameter.Parameter(
        torch.Tensor().new_empty(num_feature).cuda())
    bn_b = torch.nn.parameter.Parameter(
        torch.Tensor().new_empty(num_feature).cuda())

torch.distributed.broadcast(bn_w, 0)
torch.distributed.broadcast(bn_b, 0)

# creating DDP sync_batchnorm_layer
sbn_layer = SyncBatchNorm(num_feature).cuda()
sbn_opt = optim.SGD(sbn_layer.parameters(), lr=0.01)
sbn_layer.register_backward_hook(printgrad)
sbn_layer.weight.data = bn_w.clone()
sbn_layer.bias.data = bn_b.clone()
sbn_layer = DDP(sbn_layer)

# prepare input data for each node
data = np.random.random(data_dim)*255.0
input1 = torch.tensor(data, dtype=torch.float32).cuda()
input1.requires_grad_(True)
b_out = sbn_layer(input1)


grad = np.random.random(data_dim).astype(np.float32)*255.0
output_grad = torch.tensor(grad, dtype=torch.float32).cuda()
b_out.backward(output_grad)

# populating results/inputs to peers
world_size = torch.distributed.get_world_size()
collective_input = [torch.empty_like(input1) for i in range(world_size)]
collective_output = [torch.empty_like(b_out) for i in range(world_size)]
collective_weight_g = [torch.empty_like(
    sbn_layer.module.weight) for i in range(world_size)]
collective_bias_g = [torch.empty_like(
    sbn_layer.module.bias) for i in range(world_size)]
collective_input_g = [torch.empty_like(input1) for i in range(world_size)]
collective_output_g = [torch.empty_like(b_out) for i in range(world_size)]
torch.distributed.all_gather(collective_input, input1)
torch.distributed.all_gather(collective_output, b_out)
torch.distributed.all_gather(collective_weight_g, sbn_layer.module.weight.grad)
torch.distributed.all_gather(collective_bias_g, sbn_layer.module.bias.grad)
torch.distributed.all_gather(collective_input_g, input1.grad)
torch.distributed.all_gather(collective_output_g, output_grad)

sbn_opt.step()


if args.local_rank == 0:

    bn_layer.register_backward_hook(printgrad)
    bn_opt = optim.SGD(bn_layer.parameters(), lr=0.01)
    # concat input
    input_bn = torch.cat(collective_input, 0)
    input_bn.requires_grad_()

    # feed input
    out = bn_layer(input_bn)

    # concat output gradient and feed it to the back path
    out.backward(torch.cat(collective_output_g, 0))

    print("bn_layer, input_grad: ", input_bn.grad)
    bn_opt.step()
    print()
    print("after sync")
    print(bn_layer.state_dict())
    print(sbn_layer.state_dict())

    result = bool
    print ("-----sanity check----")
    # comparing output
    cur_res = np.allclose(out.detach().cpu().numpy(), torch.cat(
        collective_output, 0).cpu().numpy(), 1e-4, 1e-4)
    print ("compare output equal: ", cur_res)
    result = result and cur_res
    # comparing input gradient (concat for input gradient)
    cur_res = np.allclose(input_bn.grad.cpu().numpy(), torch.cat(
        collective_input_g, 0).cpu().numpy(), 1e-4, 1e-4)
    print ("compare input gradient equal: ", cur_res)
    result = result and cur_res
    # comparing bias gradient (mean for bias gradient)
    cur_res = np.allclose(bn_layer.bias.grad.cpu().numpy(), torch.mean(
        torch.stack(collective_bias_g), 0).cpu().numpy(), 1e-4, 1e-4)
    print ("compare layer parameter bias equal: ", cur_res)
    result = result and cur_res
    # comparing weight gradient (mean for weight gradient)
    cur_res = np.allclose(bn_layer.weight.grad.cpu().numpy(), torch.mean(
        torch.stack(collective_weight_g), 0).cpu().numpy(), 1e-4, 1e-4)
    print ("compare layer parameter weight equal: ", cur_res)
    result = result and cur_res
    # comparing running_var
    cur_res = np.allclose(bn_layer.running_var.cpu().numpy(
    ), sbn_layer.module.running_var.cpu().numpy(), 1e-4, 1e-4)
    print ("compare running_var equal: ", cur_res)
    result = result and cur_res
    # comparing running_mean
    cur_res = np.allclose(bn_layer.running_mean.cpu().numpy(
    ), sbn_layer.module.running_mean.cpu().numpy(), 1e-4, 1e-4)
    print ("compare running_mean equal: ", cur_res)
    result = result and cur_res
    # comparing updated weight
    cur_res = np.allclose(bn_layer.weight.detach().cpu().numpy(
    ), sbn_layer.module.weight.detach().cpu().numpy(), 1e-4, 1e-4)
    print ("compare updated layer weight equal: ", cur_res)
    result = result and cur_res
    # comparing updated bias
    cur_res = np.allclose(bn_layer.bias.detach().cpu().numpy(
    ), sbn_layer.module.bias.detach().cpu().numpy(), 1e-4, 1e-4)
    print ("compare updated layer bias equal: ", cur_res)
    result = result and cur_res

    if result:
        print("passed all test!")
    else:
        raise RuntimeError("test failed")
