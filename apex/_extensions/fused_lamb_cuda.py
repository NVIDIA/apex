import torch

from apex._custom_ops import (
    load_custom_op_library,
    scalar_float,
    scalar_int,
    tensor_list_arg,
)


load_custom_op_library("_fused_lamb_cuda", __file__)


def lamb(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    bias_correction,
    weight_decay,
    grad_averaging,
    mode,
    global_grad_norm,
    max_grad_norm,
):
    return torch.ops.apex.fused_lamb_lamb(
        scalar_int(chunk_size),
        noop_flag,
        tensor_list_arg(tensor_lists),
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        scalar_int(step),
        scalar_int(bias_correction),
        scalar_float(weight_decay),
        scalar_int(grad_averaging),
        scalar_int(mode),
        scalar_float(global_grad_norm),
        scalar_float(max_grad_norm),
    )
