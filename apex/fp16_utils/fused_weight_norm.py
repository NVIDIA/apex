import torch
from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable

class Fused_Weight_Norm(Function):
    """
    We are refactoring our fused kernels to add to Pytorch core, so that Pytorch's built-in weightnorm
    will use them transparently.  Please use Pytorch's built-in weightnorm implementation for now, to 
    future-proof your code.
    """

    @staticmethod
    def forward(ctx):
        raise NotImplementedError("Use Pytorch's built-in weightnorm implementation. "+
                                  "We are in the process of adding our fused kernels to Pytorch core, "+
                                  "so Pytorch's built-in weightnorm will use them transparently.")

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        raise NotImplementedError("Use Pytorch's built-in weightnorm implementation. "+
                                  "We are in the process of adding our fused kernels to Pytorch core, "+
                                  "so Pytorch's built-in weightnorm will use them transparently.")
