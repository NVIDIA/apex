import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int, tensor_list_arg


load_custom_op_library("_distributed_lamb_cuda", __file__)
_ops = torch.ops.apex


def multi_tensor_lamb_compute_update_term(
    chunk_size,
    noop_flag,
    tensor_lists,
    per_tensor_beta1,
    per_tensor_beta2,
    per_tensor_beta3,
    per_tensor_bias_correction,
    step,
    per_tensor_epsilon,
    mode,
    per_tensor_decay,
    global_scale,
    global_grad_norm,
    max_grad_norm,
):
    return _ops.distributed_lamb_compute_update_term(
        scalar_int(chunk_size),
        noop_flag,
        tensor_list_arg(tensor_lists),
        per_tensor_beta1,
        per_tensor_beta2,
        per_tensor_beta3,
        per_tensor_bias_correction,
        step,
        per_tensor_epsilon,
        scalar_int(mode),
        per_tensor_decay,
        global_scale,
        global_grad_norm,
        scalar_float(max_grad_norm),
    )


def multi_tensor_lamb_update_weights(
    chunk_size,
    noop_flag,
    tensor_lists,
    per_tensor_param_norm,
    per_tensor_update_norm,
    update_norm_offset,
    learning_rate,
    per_tensor_decay,
    global_grad_norm,
    use_nvlamb,
):
    return _ops.distributed_lamb_update_weights(
        scalar_int(chunk_size),
        noop_flag,
        tensor_list_arg(tensor_lists),
        per_tensor_param_norm,
        per_tensor_update_norm,
        update_norm_offset,
        learning_rate,
        per_tensor_decay,
        global_grad_norm,
        use_nvlamb,
    )
