import math
import torch
import torch.nn.modules
from sparse_inf import get_matmul_plan, get_lib_handle, sparse_matmul
from torch import Tensor
from torch import has_torch_function, handle_torch_function
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.module import Module

def sparse_linear(input, weight, bias=None, handle, plan, compressed_weights, workspace):
    r"""
        Sparse version of torch.nn.functional.linear
    """
    tens_ops = (input, weight)
    if not torch.jit.is_scripting():
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(linear, tens_ops, input, weight, bias=bias)
    if input.dim() == 2 and bias is not None:
        ret = torch.addmm(bias, input, weight.t())
    else:
        # output = input.matmul(weight.t())
        output = torch.FloatTensor((input.shape[-2], weight.t().shape[-1]), device='cuda') #TODO: same type as input, take care of stride and padding
        sparse_matmul(handle, plan, compressed_weights, workspace, input, output)

        if bias is not None:
            output += bias
        ret = output
    return ret

class SparseLinear(Module):
    r"""
        Sparse version of torch.nn.modules.Linear.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    # initialize the opaque data structures and pointers
    self.compressed_weights = torch.ByteTensor(0)
    self.workspace = torch.ByteTensor(0)
    self.plan = None
    self.handle = None
    self.first_call = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.first_call:
            self.handle = torch.ByteTensor(get_lib_handle())
            nrw, ncw = weight.shape
            _, nri, nci = input.shape
            self.plan = torch.ByteTensor(get_matmul_plan(self.weight, input, nrw, ncw, nri, nci, self.workspace, self.compressed_weights))
            self.first_call = False
        return sparse_linear(input, self.weight, self.bias, self.handle, self.plan, self.compressed_weights, self.workspace)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )