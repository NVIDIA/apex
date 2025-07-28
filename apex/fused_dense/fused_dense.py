import torch
from torch import nn
import fused_dense_cuda
from apex._autocast_utils import _cast_if_autocast_enabled
import math 

#implements fused GEMM+bias in forward pass using mlp_cuda from apex
class FusedDenseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        output = fused_dense_cuda.linear_bias_forward(input, weight, bias.t())
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
    def forward(ctx, input, weight, bias, weight2, bias2):
        '''
        The forward method of the FusedDenseGELUDense layer performs the following operations:
            Applies the first dense layer (dense1) to the input tensor.
            Applies the GELU activation function (act) to the result.
            Applies the second dense layer (dense2) to the GELU-activated output.
        '''
        ctx.save_for_backward(input, weight, weight2)
        output, output2, gelu = fused_dense_cuda.linear_gelu_linear_forward(input, weight, bias, weight2, bias2)
        ctx.save_for_backward(input, weight, weight2, gelu, output)
        return output2

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight2, gelu, output = ctx.saved_tensors
        grad_input, grad_weight, grad_bias, grad_weight2, grad_bias2 = fused_dense_cuda.linear_gelu_linear_backward(input, gelu, output, weight, weight2, grad_output)
        return grad_input, grad_weight, grad_bias, grad_weight2, grad_bias2

def fused_dense_function(input, weight, bias):
    args = _cast_if_autocast_enabled(input, weight, bias)
    with torch.amp.autocast('cuda', enabled=False):
        return FusedDenseFunc.apply(*args)

def dense_no_bias_function(input, weight):
    args = _cast_if_autocast_enabled(input, weight)
    with torch.amp.autocast('cuda', enabled=False):
        return DenseNoBiasFunc.apply(*args)

def fused_dense_gelu_dense_function(input, weight1, bias1, weight2, bias2):
    args = _cast_if_autocast_enabled(input, weight1, bias1, weight2, bias2)
    with torch.amp.autocast('cuda', enabled=False):
        return FusedDenseGeluDenseFunc.apply(*args)

class FusedDense(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FusedDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            #assert False, "no-bias option not added yet"
            self.register_parameter('bias', None)
        self.reset_parameters()


    def forward(self, input):
        if self.bias is not None:
            return fused_dense_function(input, self.weight, self.bias)
        else:
            return dense_no_bias_function(input, self.weight)
        

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
#======================================================================================= 
# 
#======================================================================================= 
class FusedDenseGeluDense(nn.Module):
    '''
    https://zeta.apac.ai/en/latest/zeta/nn/modules/fused_gelu_dense/
    module combines dense layers with GELU activations in a single neural network layer.
    layer consists of two dense sub-layers, each followed by a GELU activation function. 
    It takes an input tensor and passes it through these sub-layers to produce the final output.
    Parameters:
        dim (int): Input dimension.
        dim_out (int): Output dimension.
        bias (bool, optional): Whether to include bias terms. Defaults to True.
        has_fp16_weights (bool, optional): Whether to use fp16 weights. Defaults to False.
        threshold (float, optional): Threshold for quantization. Defaults to 6.0.

    layer consists of the following internal layers:
        dense1: The first dense layer.
        act: The GELU activation function.
        dense2: The second dense layer.

    '''
    def __init__(self, in_features, intermediate_features, out_features, bias=True):
        super(FusedDenseGeluDense, self).__init__()
        assert bias == True, "DenseGeluDense module without bias is currently not supported"
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.randn(intermediate_features, in_features))
        self.bias1 = nn.Parameter(torch.randn(intermediate_features))
        self.weight2 = nn.Parameter(torch.randn(out_features, intermediate_features))
        self.bias2 = nn.Parameter(torch.randn(out_features))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias1, -bound, bound)
        if self.bias2 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias2, -bound, bound)


    def forward(self, input):
        return fused_dense_gelu_dense_function(input, self.weight1, self.bias1, self.weight2, self.bias2)

