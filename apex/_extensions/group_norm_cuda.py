import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_group_norm_cuda", __file__)


def forward(input, groups, weight, bias, eps, passes, with_swish=False):
    return torch.ops.apex.group_norm_forward(
        input,
        scalar_int(groups),
        weight,
        bias,
        scalar_float(eps),
        scalar_int(passes),
        bool(with_swish),
    )


def backward(grad_output, sums, input, groups, weight, bias, eps, passes, with_swish=False):
    return torch.ops.apex.group_norm_backward(
        grad_output,
        sums,
        input,
        scalar_int(groups),
        weight,
        bias,
        scalar_float(eps),
        scalar_int(passes),
        bool(with_swish),
    )
