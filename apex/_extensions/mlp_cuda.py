import torch

from apex._custom_ops import load_custom_op_library


load_custom_op_library("_mlp_cuda", __file__)


def forward(use_bias, activation, inputs):
    return torch.ops.apex.mlp_forward(use_bias, activation, list(inputs))


def backward(use_bias, activation, grad_o, fprop_outputs, inputs):
    return torch.ops.apex.mlp_backward(
        use_bias, activation, grad_o, list(fprop_outputs), list(inputs)
    )
