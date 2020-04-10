from copy import copy
import math
import torch
from torch import nn
import mlp_cuda
from .. import amp

class MlpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        output = mlp_cuda.forward(args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        return output[0]

    @staticmethod
    def backward(ctx, grad_o):
        grads = mlp_cuda.backward(grad_o, ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return tuple(grads)

mlp_function = amp.half_function(MlpFunction.apply)

class MLP(torch.nn.Module):
    """Launch MLP in C++

    Args:
        mlp_sizes (list of int): MLP sizes. Example: [1024,1024,1024] will create 2 MLP layers with shape 1024x1024
        bias (bool): Default True:
        relu (bool): Default True
    """
    def __init__(self, mlp_sizes, bias=True, relu=True):
        if not (bias and relu):
            raise TypeError("bias and relu must be both true.")
        super(MLP, self).__init__()
        self.num_layers = len(mlp_sizes) - 1
        self.mlp_sizes = copy(mlp_sizes)
        self.bias = bias
        self.relu= relu

        # ignoring bias = False now
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = torch.nn.Parameter(torch.empty(mlp_sizes[i+1], mlp_sizes[i]))
            self.weights.append(w)
            name = 'weight_{}'.format(i)
            setattr(self, name, w)
            b = torch.nn.Parameter(torch.empty(mlp_sizes[i+1]))
            self.biases.append(b)
            name = 'bias_{}'.format(i)
            setattr(self, name, b)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            dimsum = weight.size(0) + weight.size(1)
            std = math.sqrt(2. / float(dimsum))
            nn.init.normal_(weight, 0., std)
        for bias in self.biases:
            std = math.sqrt(1. / float(bias.size(0)))
            nn.init.normal_(bias, 0., std)

    def forward(self, input):
        return mlp_function(input, *self.weights, *self.biases)

    def extra_repr(self):
        s = F"MLP sizes: {self.mlp_sizes}, Bias={self.bias}, ReLU={self.relu}"
        return s
