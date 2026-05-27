import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int, tensor_list_arg


load_custom_op_library("_distributed_adam_cuda", __file__)
_ops = torch.ops.apex


def multi_tensor_fused_adam(
    chunk_size,
    noop_flag,
    tensor_lists,
    grad_scale,
    lr,
    beta1,
    beta2,
    eps,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    return _ops.distributed_adam_multi_tensor_fused_adam(
        scalar_int(chunk_size),
        noop_flag,
        tensor_list_arg(tensor_lists),
        grad_scale,
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(eps),
        scalar_int(step),
        scalar_int(mode),
        scalar_int(bias_correction),
        scalar_float(weight_decay),
    )


def multi_tensor_fused_adam_capturable(
    chunk_size,
    noop_flag,
    tensor_lists,
    grad_scale,
    lr,
    beta1,
    beta2,
    eps,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    return _ops.distributed_adam_multi_tensor_fused_adam_capturable(
        scalar_int(chunk_size),
        noop_flag,
        tensor_list_arg(tensor_lists),
        grad_scale,
        lr,
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(eps),
        step,
        scalar_int(mode),
        scalar_int(bias_correction),
        scalar_float(weight_decay),
    )


def multi_tensor_fused_adam_with_param_remainders(
    chunk_size,
    noop_flag,
    tensor_lists,
    grad_scale,
    lr,
    beta1,
    beta2,
    eps,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    return _ops.distributed_adam_multi_tensor_fused_adam_with_param_remainders(
        scalar_int(chunk_size),
        noop_flag,
        tensor_list_arg(tensor_lists),
        grad_scale,
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(eps),
        scalar_int(step),
        scalar_int(mode),
        scalar_int(bias_correction),
        scalar_float(weight_decay),
    )
