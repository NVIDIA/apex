import torch
import torch.nn as nn
import argparse
import os

from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import SyncBatchNorm
import numpy as np
import torch.optim as optim

PRECISION = 1e-3

def compare(desc, inp1, inp2):
    close = np.allclose(inp1.detach().cpu().numpy(),inp2.detach().cpu().numpy(), PRECISION, PRECISION)
    if not close:
        print(desc, close)
        print('d1: ', inp1.clone().detach().cpu().numpy() - inp2.clone().detach().cpu().numpy())
        print('d2: ', inp2.clone().detach().cpu().numpy())
    return close

def getBatchNormModule(module):
    for m in module.modules():
       if isinstance(m, (torch.nn.BatchNorm3d, SyncBatchNorm)):
           return m

def compareLayers(layer1, layer2):
    result = True
    # comparing running_var
    result = (result and
              compare("compare running_var equal: ",
                      layer1.running_var,
                      layer2.running_var
              )
    )
    # comparing running_mean
    result = (result and
              compare("compare running_mean equal: ",
                      layer1.running_mean,
                      layer2.running_mean
              )
    )
    # comparing updated weight
    result = (result and
              compare("compare layer weight equal: ",
                      layer1.weight,
                      layer2.weight
              )
    )
    # comparing updated bias
    result = (result and
              compare("compare layer bias equal: ",
                      layer1.bias,
                      layer2.bias
              )
    )
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--fp16", action='store_true', default=False)
args = parser.parse_args()

if args.fp16:
    PRECISION = 1e-1

args.world_size = int(os.environ['WORLD_SIZE'])

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

torch.manual_seed(1)

num_feature = 8
num_batch = 2
num_iteration = 5
data_dim = (num_iteration, num_batch, num_feature, 20, 8, 1)

bn_layer = None
sbn_layer = None
bn_w = None
bn_b = None
bn_m = None
bn_v = None

# creating torch.nn.BatchNorm layer (ground truth)
bn_layer = nn.BatchNorm3d(num_feature).cuda()

bn_layer.running_mean.random_(0, 100)
bn_layer.running_var.random_(0, 100)
bn_layer.bias.data.random_(0, 100)
bn_layer.weight.data.random_(0, 100)

bn_w = bn_layer.weight
bn_b = bn_layer.bias
bn_m = bn_layer.running_mean
bn_v = bn_layer.running_var
bn_opt = optim.SGD(bn_layer.parameters(), lr=1.0)

# creating DDP sync_batchnorm_layer
sbn_layer = SyncBatchNorm(num_feature).cuda()

sbn_layer.weight.data = bn_w.clone()
sbn_layer.bias.data = bn_b.clone()
sbn_layer.running_mean.data = bn_m.clone()
sbn_layer.running_var.data = bn_v.clone()

sbn_opt = optim.SGD(sbn_layer.parameters(), lr=1.0*args.world_size)

sbn_layer = DDP(sbn_layer)

# reset seed ( because DP sync bn is only created on rank 0)
torch.manual_seed(100)

data = torch.randn(*data_dim).cuda()
grad = torch.randn(*data_dim).cuda()

if args.fp16:
    sbn_layer_half = SyncBatchNorm(num_feature).cuda()
    sbn_layer_half.load_state_dict(bn_layer.state_dict())
    sbn_layer_half = sbn_layer_half.half()
    sbn_opt_half = optim.SGD(sbn_layer_half.parameters(), lr=1.0*args.world_size)
    sbn_layer_half = DDP(sbn_layer_half)
    data_half = data.half()
    grad_half = grad.half()

    bn_layer_half = nn.BatchNorm3d(num_feature).cuda().half()
    bn_layer_half.load_state_dict(bn_layer.state_dict())
    bn_opt_half = optim.SGD(bn_layer_half.parameters(), lr=1.0)


def execute(layer, inp, grad=None, is_training=True):
    layer.train(mode=is_training)
    inp = inp.clone().detach().requires_grad_(True)
    grad = grad.clone().detach().requires_grad_(False)
    out = layer(inp)
    if is_training:
        out.backward(grad)
    return inp, out

training = True
inference = True
test_result = True
start = args.local_rank * num_batch//args.world_size
finish = (args.local_rank + 1) * num_batch//args.world_size
enabling_fp16_syncbn = True
enabling_fp16_bn = True

for i in range(0, num_iteration):
    result = True

    # swap training and inference
    if inference and (i+1) % 3 == 0:
        training = not training
    cur_input = data[i]
    cur_grad = grad[i]
    if args.fp16:
        cur_input_half = data_half[i]
        cur_grad_half = grad_half[i]

    bn_inp, bn_out = execute(bn_layer, cur_input, cur_grad, is_training=training)

    if sbn_layer:
        sbn_inp, sbn_out = execute(sbn_layer, cur_input[start:finish], cur_grad[start:finish], is_training=training)
        result = (result and compareLayers(bn_layer, getBatchNormModule(sbn_layer)))
        result = (result and 
                  compare("compare training output bn vs apex sync bn: ",
                          sbn_out.clone().detach_(), bn_out[start:finish].clone().detach_()
                  )
        )
        if training:
            result = (result and 
                      compare("compare training input grad bn vs apex sync bn: ",
                              sbn_inp.grad, bn_inp.grad[start:finish]
                      )
            )
        if args.fp16:
            if enabling_fp16_syncbn: 
                sbn_inp_half, sbn_out_half = execute(sbn_layer_half, cur_input_half[start:finish], cur_grad_half[start:finish], is_training=training)
                result = (result and compareLayers(getBatchNormModule(sbn_layer), getBatchNormModule(sbn_layer_half)))
                result = (result and 
                          compare("compare training output apex fp16 vs fp32: ",
                                  sbn_out.clone().detach_(), sbn_out_half.clone().detach_()
                          )
                )
            if args.local_rank == 0 and enabling_fp16_bn:
                bn_inp_half, bn_out_half = execute(bn_layer_half, cur_input_half, cur_grad_half, is_training=training)
                result = (result and compareLayers(bn_layer, bn_layer_half))
                result = (result and 
                          compare("compare training output pytorch fp16 vs fp32: ",
                                  bn_out.clone().detach_(), bn_out_half.clone().detach_()
                          )
                )
            if training:
                if enabling_fp16_syncbn: 
                    result = (result and 
                              compare("compare training input grad bn vs apex sync bn: ",
                                      sbn_inp.grad, sbn_inp_half.grad
                              )
                    )
                if args.local_rank == 0 and enabling_fp16_bn:
                    result = (result and 
                              compare("compare training input grad bn vs apex sync bn: ",
                                      bn_inp.grad, bn_inp_half.grad
                              )
                    )

    if not result:
        print("\n====sbn failed at iter: {0}, training? {1}====\n".format(i, training))
        test_result = test_result and result
        result = True

    if training:
        bn_opt.step()

        if sbn_layer:
            sbn_opt.step()
            result = (result and compareLayers(bn_layer, getBatchNormModule(sbn_layer)))

        if args.fp16:
            if enabling_fp16_syncbn: 
                sbn_opt_half.step()
                result = (result and compareLayers(getBatchNormModule(sbn_layer), getBatchNormModule(sbn_layer_half)))
            if args.local_rank == 0 and enabling_fp16_bn:
                bn_opt_half.step()
                result = (result and compareLayers(bn_layer, bn_layer_half))

    if not result and args.local_rank == 0:
        print("\n====training failed at iter: {0}, training? {1}====\n".format(i, training))

    test_result = test_result and result


if test_result:
    print("passed all test!")
else:
    raise RuntimeError("test failed")
