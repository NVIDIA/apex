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
        grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_backward(input, weight, grad_output)
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


class FusedDenseGeluDenseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2):
        ctx.save_for_backward(input, weight1, weight2)
        output1, output2, gelu_in = fused_dense_cuda.linear_gelu_linear_forward(input, weight1, bias1, weight2, bias2)
        ctx.save_for_backward(input, weight1, weight2, gelu_in, output1)
        return output2

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, weight2, gelu_in, output1 = ctx.saved_tensors
        grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_gelu_linear_backward(input, gelu_in, output1, weight1, weight2, grad_output)
        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2


fused_dense_function = amp.half_function(FusedDenseFunc.apply)
dense_no_bias_function = amp.half_function(DenseNoBiasFunc.apply)
fused_dense_gelu_dense_function = amp.half_function(FusedDenseGeluDenseFunc.apply)

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

class FusedDenseGeluDense(nn.Module):
    def __init__(self, in_features, intermediate_features, out_features, bias=True):
        super(FusedDenseGeluDense, self).__init__()
        assert bias == True, "DenseGeluDense module without bias is currently not supported"
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(intermediate_features, in_features))
        self.bias1 = nn.Parameter(torch.Tensor(intermediate_features))
        self.weight2 = nn.Parameter(torch.Tensor(out_features, intermediate_features))
        self.bias2 = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return fused_dense_gelu_dense_function(input, self.weight1, self.bias1, self.weight2, self.bias2)

