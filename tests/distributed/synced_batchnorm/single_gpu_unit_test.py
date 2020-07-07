import torch
import numpy as np
import apex
if True:
    print("using setup tools")
    import syncbn
else:
    print("using jit")
    from torch.utils.cpp_extension import load
    syncbn = load(name='syncbn', sources=['../../csrc/syncbn.cpp', '../../csrc/welford.cu'])

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
space_size = 16
batch_size = 5


error = 1e-5

np.random.seed(1)
dtype = np.float32
inp = (np.random.randn(batch_size, feature_size, space_size, space_size)).astype(dtype)
grad = (np.random.randn(batch_size, feature_size, space_size, space_size)).astype(dtype)
weight = (np.random.randn(feature_size)).astype(dtype)
bias = (np.random.randn(feature_size)).astype(dtype)
count = torch.cuda.IntTensor([batch_size*space_size**2])

type_tensor = torch.cuda.FloatTensor
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

#mean, var, var_biased = syncbn.welford_mean_var(inp_t)
mean, var_biased = syncbn.welford_mean_var(inp_t)
inv_std = 1.0 / torch.sqrt(var_biased + eps)

bn = torch.nn.BatchNorm2d(feature_size).cuda()
bn.momentum = 1.0
bn.weight.data = weight_t.clone()
bn.bias.data = bias_t.clone()
inp_bn = inp_t.clone().requires_grad_()
grad_bn = grad_output_t.clone().detach()
out_bn = bn(inp_bn)
out_bn.backward(grad_bn)

sbn = apex.parallel.SyncBatchNorm(feature_size).cuda()
sbn.momentum = 1.0
sbn.weight.data = weight_t.clone()
sbn.bias.data = bias_t.clone()
inp_sbn = inp_t.clone().requires_grad_()
grad_sbn = grad_output_t.clone().detach()
out_sbn = sbn(inp_sbn)
out_sbn.backward(grad_sbn)

sbn_c_last = apex.parallel.SyncBatchNorm(feature_size, channel_last=True).cuda()
sbn_c_last.momentum = 1.0
sbn_c_last.weight.data = weight_t.clone()
sbn_c_last.bias.data = bias_t.clone()
inp_sbn_c_last = inp_t.clone().transpose(-1, 1).contiguous().requires_grad_()
grad_sbn_c_last = grad_output_t.clone().transpose(-1, 1).contiguous().detach()
out_sbn_c_last = sbn_c_last(inp_sbn_c_last)
out_sbn_c_last.backward(grad_sbn_c_last)

sbn_result = True
sbn_result_c_last = True
bn_result = True

sbn_result = compare("comparing mean: ", mean, m, error) and sbn_result
#sbn_result = compare("comparing variance: ", var, unb_v, error) and sbn_result
sbn_result = compare("comparing biased variance: ", var_biased, b_v, error) and sbn_result


out = syncbn.batchnorm_forward(inp_t, mean, inv_std, weight_t, bias_t)
out_r = weight_r * (inp2_r - m.view(-1, 1, 1)) * torch.rsqrt(b_v.view(-1,1,1) + eps) + bias_r

sbn_result = compare("comparing output: ", out, out_r, error) and sbn_result
compare("comparing bn output: ", out_bn, out_r, error)

grad_output_t = type_tensor(grad)

grad_output_r = ref_tensor(grad.transpose(1, 0, 2, 3).reshape(feature_size, -1))
grad_output2_r = ref_tensor(grad)

grad_bias_r = grad_output_r.sum(1)
grad_weight_r = ((inp2_r - m.view(-1, 1, 1)) * torch.rsqrt(b_v.view(-1,1,1) + eps) * grad_output2_r).transpose(1,0).contiguous().view(feature_size, -1).sum(1)

sum_dy_r = grad_output_r.sum(1)
mean_dy_r = grad_output_r.mean(1)
sum_dy_xmu_r = ((inp2_r - m.view(-1, 1, 1)) * grad_output2_r).transpose(1,0).contiguous().view(feature_size, -1).sum(1)
mean_dy_xmu_r = ((inp2_r - m.view(-1, 1, 1)) * grad_output2_r).transpose(1,0).contiguous().view(feature_size, -1).mean(1)

grad_input_r = (grad_output2_r - mean_dy_r.view(-1, 1, 1) - (inp2_r - m.view(-1, 1, 1)) / (b_v.view(-1,1,1) + eps) * mean_dy_xmu_r.view(-1, 1, 1) ) * torch.rsqrt(b_v.view(-1,1,1) + eps) * weight_r.view(-1,1,1)

sum_dy, sum_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn(grad_output_t, inp_t, mean, inv_std, weight_t)
grad_input = syncbn.batchnorm_backward(grad_output_t, inp_t, mean, inv_std, weight_t, sum_dy, sum_dy_xmu, count)
sbn_result = compare("comparing bias grad: ", grad_bias, grad_bias_r, error) and sbn_result
sbn_result = compare("comparing weight grad: ", grad_weight, grad_weight_r, error) and sbn_result
sbn_result = compare("comparing sum_dy grad: ", sum_dy, sum_dy_r, error) and sbn_result
sbn_result = compare("comparing sum_dy_xmu grad: ", sum_dy_xmu, sum_dy_xmu_r, error) and sbn_result
sbn_result = compare("comparing input grad: ", grad_input, grad_input_r, error) and sbn_result
compare("comparing bn input grad: ", inp_bn.grad, grad_input_r, error)
sbn_result = compare("comparing sbn input grad: ", inp_sbn.grad, grad_input_r, error) and sbn_result

compare("comparing bn/sbn output: ", out_bn, out_sbn, error)
sbn_result = compare("comparing running_mean: ", bn.running_mean.data, sbn.running_mean.data, error) and sbn_result
sbn_result = compare("comparing running_variance: ", bn.running_var.data, sbn.running_var.data, error) and sbn_result
compare("comparing grad_input: ", inp_bn.grad, inp_sbn.grad, error)
compare("comparing grad_bias: ", bn.bias.grad, sbn.bias.grad, error)
compare("comparing grad_bias bn to ref: ", bn.bias.grad, grad_bias_r, error)
sbn_result = compare("comparing grad_bias sbn to ref: ", sbn.bias.grad, grad_bias_r, error) and sbn_result
compare("comparing grad_weight: ", bn.weight.grad, sbn.weight.grad, error)
compare("comparing grad_weight bn to ref: ", bn.weight.grad, grad_weight_r, error)
sbn_result = compare("comparing grad_weight sbn to ref: ", sbn.weight.grad, grad_weight_r, error) and sbn_result

compare("comparing channel last bn/sbn output: ", out_bn, out_sbn_c_last.transpose(-1, 1).contiguous(), error)
sbn_result_c_last = compare("comparing channel last running_mean: ", bn.running_mean.data, sbn_c_last.running_mean.data, error) and sbn_result_c_last
sbn_result_c_last = compare("comparing channel last running_variance: ", bn.running_var.data, sbn_c_last.running_var.data, error) and sbn_result_c_last
compare("comparing channel last grad_input: ", inp_bn.grad, inp_sbn_c_last.grad.transpose(-1, 1).contiguous(), error)
compare("comparing channel last grad_bias: ", bn.bias.grad, sbn_c_last.bias.grad, error)
sbn_result_c_last = compare("comparing channel last grad_bias sbn to ref: ", sbn_c_last.bias.grad, grad_bias_r, error) and sbn_result_c_last
compare("comparing channel last grad_weight: ", bn.weight.grad, sbn_c_last.weight.grad, error)
sbn_result_c_last = compare("comparing channel last grad_weight sbn to ref: ", sbn_c_last.weight.grad, grad_weight_r, error) and sbn_result_c_last

if sbn_result:
    print("====SBN single gpu passed tests")
else:
    print("*SBN single gpu failed*")

if sbn_result_c_last:
    print("====SBN channel last single gpu passed tests")
else:
    print("*SBN channel last single gpu failed*")
