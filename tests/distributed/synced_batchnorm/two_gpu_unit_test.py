import torch
import numpy as np
import apex
import syncbn
import os
import argparse
import torch.optim as optim

def compare(desc, inp1, inp2, error):
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

feature_size = 10
space_size = 40
batch_size = 32


from apex.parallel import DistributedDataParallel as DDP
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--fp16", action='store_true', default=False)
parser.add_argument("--fp64", action='store_true', default=False)
args = parser.parse_args()
args.world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
start = args.local_rank * batch_size//args.world_size
finish = (args.local_rank + 1) * batch_size//args.world_size

error = 1e-5
dtype = np.float32
if args.fp16:
    error = 1e-3
    dtype = np.float16
elif args.fp64:
    error = 1e-8
    dtype = np.float64

np.random.seed(18)
inp = np.random.randn(batch_size, feature_size, space_size, space_size).astype(dtype)
grad = np.random.randn(batch_size, feature_size, space_size, space_size).astype(dtype)
weight = np.random.randn(feature_size).astype(dtype)
bias = np.random.randn(feature_size).astype(dtype)


type_tensor = torch.cuda.FloatTensor
if args.fp16:
    type_tensor = torch.cuda.HalfTensor
if args.fp64:
    type_tensor = torch.cuda.DoubleTensor

ref_tensor = torch.cuda.DoubleTensor

inp_t = type_tensor(inp)
weight_t = type_tensor(weight)
bias_t = type_tensor(bias)

inp_r = ref_tensor(inp.transpose(1, 0, 2, 3).reshape(feature_size, -1))
inp2_r = ref_tensor(inp)
weight_r = ref_tensor(weight).view(-1, 1, 1)
bias_r = ref_tensor(bias).view(-1, 1, 1)

grad_output_t = type_tensor(grad)

m = inp_r.mean(1)
b_v = inp_r.var(1, unbiased=False)
unb_v = inp_r.var(1, unbiased=True)

eps = 1e-5

mean, var_biased = syncbn.welford_mean_var(inp_t)
inv_std = 1.0 / torch.sqrt(var_biased + eps)

bn = torch.nn.BatchNorm2d(feature_size).cuda()
bn.momentum = 1.0
bn.weight.data = weight_t.clone()
bn.bias.data = bias_t.clone()
if args.fp16:
    bn.half()
if args.fp64:
    bn.double()
inp_bn = inp_t.clone().requires_grad_()
grad_bn = grad_output_t.clone().detach()
out_bn = bn(inp_bn)
out_bn.backward(grad_bn)
# compensating the averaging over processes done by DDP
# in order to produce mathematically equivalent result
# https://github.com/NVIDIA/apex/issues/134#issuecomment-458307368
for param in bn.parameters():
    param.grad = param.grad / args.world_size
bn_opt = optim.SGD(bn.parameters(), lr=1.0)

sbn = apex.parallel.SyncBatchNorm(feature_size).cuda()
sbn.momentum = 1.0
sbn.weight.data = weight_t.clone()
sbn.bias.data = bias_t.clone()
if args.fp16:
    sbn.half()
if args.fp64:
    sbn.double()
sbn = DDP(sbn)
sbn_opt = optim.SGD(sbn.parameters(), lr=1.0)
inp_sbn = inp_t.clone().requires_grad_()
grad_sbn = grad_output_t.clone().detach()
out_sbn = sbn(inp_sbn[start:finish])
out_sbn.backward(grad_sbn[start:finish])

sbn_result = True
bn_result = True

if args.local_rank == 0:
    sbn_result = compare("comparing mean: ", mean, m, error) and sbn_result
    sbn_result = compare("comparing biased variance: ", var_biased, b_v, error) and sbn_result

out = syncbn.batchnorm_forward(inp_t, mean, inv_std, weight_t, bias_t)
out_r = weight_r * (inp2_r - m.view(-1, 1, 1)) * torch.rsqrt(b_v.view(-1,1,1) + eps) + bias_r

if args.local_rank == 0:
    sbn_result = compare("comparing output: ", out, out_r, error) and sbn_result
    compare("comparing bn output: ", out_bn, out_r, error)

grad_output_t = type_tensor(grad)

grad_output_r = ref_tensor(grad.transpose(1, 0, 2, 3).reshape(feature_size, -1))
grad_output2_r = ref_tensor(grad)

grad_bias_r = grad_output_r.sum(1)
grad_weight_r = ((inp2_r - m.view(-1, 1, 1)) * torch.rsqrt(b_v.view(-1,1,1) + eps) * grad_output2_r).transpose(1,0).contiguous().view(feature_size, -1).sum(1)

mean_dy_r = grad_output_r.mean(1)
mean_dy_xmu_r = ((inp2_r - m.view(-1, 1, 1)) * grad_output2_r).transpose(1,0).contiguous().view(feature_size, -1).mean(1)

grad_input_r = (grad_output2_r - mean_dy_r.view(-1, 1, 1) - (inp2_r - m.view(-1, 1, 1)) / (b_v.view(-1,1,1) + eps) * mean_dy_xmu_r.view(-1, 1, 1) ) * torch.rsqrt(b_v.view(-1,1,1) + eps) * weight_r.view(-1,1,1)

mean_dy, mean_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn(grad_output_t, inp_t, mean, inv_std, weight_t)
grad_input = syncbn.batchnorm_backward(grad_output_t, inp_t, mean, inv_std, weight_t, mean_dy, mean_dy_xmu)
if args.local_rank == 0:
    sbn_result = compare("comparing bias grad: ", grad_bias, grad_bias_r, error) and sbn_result
    sbn_result = compare("comparing weight grad: ", grad_weight, grad_weight_r, error) and sbn_result
    sbn_result = compare("comparing mean_dy grad: ", mean_dy, mean_dy_r, error) and sbn_result
    sbn_result = compare("comparing mean_dy_xmu grad: ", mean_dy_xmu, mean_dy_xmu_r, error) and sbn_result
    sbn_result = compare("comparing input grad: ", grad_input, grad_input_r, error) and sbn_result
    compare("comparing bn input grad: ", inp_bn.grad, grad_input_r, error)

if args.local_rank == 0:
    sbn_result = compare("comparing running_mean: ", bn.running_mean.data, sbn.module.running_mean.data, error) and sbn_result
    sbn_result = compare("comparing running_variance: ", bn.running_var.data, sbn.module.running_var.data, error) and sbn_result

# execute by both
compare("comparing layers output: ", out_bn[start:finish], out_sbn, error) and sbn_result
compare("comparing layers grad_input: ", inp_bn.grad[start:finish], inp_sbn.grad[start:finish], error) and sbn_result

bn_opt.step()
sbn_opt.step()

if args.local_rank == 0:
    compare("comparing bn vs sbn bias: ", bn.bias, sbn.module.bias, error)
    compare("comparing bn vs sbn weight: ", bn.weight, sbn.module.weight, error)


if sbn_result:
    print("====SBN two gpu passed tests")
else:
    print("*SBN two gpu failed*")
