import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int, tensor_list_arg


load_custom_op_library("_fused_adam_cuda", __file__)


def strided_check_finite(overflow_flag, p_copy, stride, clear_overflow_first):
    return torch.ops.apex.fused_adam_strided_check_finite(
        overflow_flag, p_copy, scalar_int(stride), scalar_int(clear_overflow_first)
    )


def adam(
    p,
    p_copy,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    return torch.ops.apex.fused_adam_adam(
        p,
        p_copy,
        m,
        v,
        g,
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(eps),
        scalar_float(grad_scale),
        scalar_int(step),
        scalar_int(mode),
        scalar_int(bias_correction),
        scalar_float(decay),
    )


def reversible_adam(
    p,
    p_copy,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    return torch.ops.apex.fused_adam_reversible_adam(
        p,
        p_copy,
        m,
        v,
        g,
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(eps),
        scalar_float(grad_scale),
        scalar_int(step),
        scalar_int(mode),
        scalar_int(bias_correction),
        scalar_float(decay),
    )


def adam_mt(
    chunk_size,
    overflow_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    return torch.ops.apex.fused_adam_adam_mt(
        scalar_int(chunk_size),
        overflow_flag,
        tensor_list_arg(tensor_lists),
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(eps),
        scalar_float(grad_scale),
        scalar_int(step),
        scalar_int(mode),
        scalar_int(bias_correction),
        scalar_float(decay),
    )


def maybe_adam_undo(
    overflow_flag,
    p,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    return torch.ops.apex.fused_adam_maybe_adam_undo(
        overflow_flag,
        p,
        m,
        v,
        g,
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(eps),
        scalar_float(grad_scale),
        scalar_int(step),
        scalar_int(mode),
        scalar_int(bias_correction),
        scalar_float(decay),
    )


def maybe_cast(overflow_flag, p_in, p_out):
    return torch.ops.apex.fused_adam_maybe_cast(overflow_flag, p_in, p_out)


def maybe_cast_mt(chunk_size, overflow_flag, tensor_lists):
    return torch.ops.apex.fused_adam_maybe_cast_mt(
        scalar_int(chunk_size), overflow_flag, tensor_list_arg(tensor_lists)
    )
