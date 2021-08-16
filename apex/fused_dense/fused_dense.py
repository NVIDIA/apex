import torch
from torch import nn
import fused_dense_cuda
from .. import amp
#implements fused GEMM+bias in forward pass using mlp_cuda from apex
class FusedDenseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        output = fused_dense_cuda.linear_bias_forward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_b = fused_dense_cuda.linear_bias_backward(input, weight, grad_output)
        return grad_input, grad_weight, grad_bias

class DenseNoBiasFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight


fused_dense_function = amp.half_function(FusedDenseFunc.apply)
dense_no_bias_function = amp.half_function(DenseNoBiasFunc.apply)

class FusedDense(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FusedDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            #assert False, "no-bias option not added yet"
            self.register_parameter('bias', None)

    def forward(self, input):
        if self.bias is not None:
            return fused_dense_function(input, self.weight, self.bias)
        else:
            return dense_no_bias_function(input, self.weight)


