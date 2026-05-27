import torch

from apex._custom_ops import load_custom_op_library, scalar_float


load_custom_op_library("_scaled_upper_triang_masked_softmax_cuda", __file__)


def forward(input, scale_factor):
    return torch.ops.apex.scaled_upper_triang_masked_softmax_forward(
        input, scalar_float(scale_factor)
    )


def backward(output_grads, softmax_results, scale_factor):
    return torch.ops.apex.scaled_upper_triang_masked_softmax_backward(
        output_grads, softmax_results, scalar_float(scale_factor)
    )
