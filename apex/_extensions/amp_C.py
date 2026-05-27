import torch

from apex._custom_ops import load_custom_op_library, scalar_float, tensor_list_arg


load_custom_op_library("_amp_C", __file__)
_ops = torch.ops.apex


def multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
    return _ops.amp_multi_tensor_scale(chunk_size, noop_flag, tensor_list_arg(tensor_lists), scalar_float(scale))


def multi_tensor_sgd(
    chunk_size,
    noop_flag,
    tensor_lists,
    wd,
    momentum,
    dampening,
    lr,
    nesterov,
    first_run,
    wd_after_momentum,
    scale,
):
    return _ops.amp_multi_tensor_sgd(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        scalar_float(wd),
        scalar_float(momentum),
        scalar_float(dampening),
        scalar_float(lr),
        nesterov,
        first_run,
        wd_after_momentum,
        scalar_float(scale),
    )


def multi_tensor_axpby(chunk_size, noop_flag, tensor_lists, a, b, arg_to_check):
    return _ops.amp_multi_tensor_axpby(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        scalar_float(a),
        scalar_float(b),
        arg_to_check,
    )


def multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor_python=None):
    return _ops.amp_multi_tensor_l2norm(chunk_size, noop_flag, tensor_list_arg(tensor_lists), per_tensor_python)


def multi_tensor_l2norm_mp(chunk_size, noop_flag, tensor_lists, per_tensor_python=None):
    return _ops.amp_multi_tensor_l2norm_mp(chunk_size, noop_flag, tensor_list_arg(tensor_lists), per_tensor_python)


def multi_tensor_l2norm_scale(chunk_size, noop_flag, tensor_lists, scale, per_tensor_python=None):
    return _ops.amp_multi_tensor_l2norm_scale(
        chunk_size, noop_flag, tensor_list_arg(tensor_lists), scalar_float(scale), per_tensor_python
    )


def multi_tensor_unscale_l2norm(chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor_python=None):
    return _ops.amp_multi_tensor_unscale_l2norm(
        chunk_size, noop_flag, tensor_list_arg(tensor_lists), inv_scale, per_tensor_python
    )


def multi_tensor_lamb_stage1_cuda(
    chunk_size,
    noop_flag,
    tensor_lists,
    per_tensor_decay,
    step,
    beta1,
    beta2,
    epsilon,
    global_grad_norm,
    max_global_grad_norm,
):
    return _ops.amp_multi_tensor_lamb_stage1_cuda(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        per_tensor_decay,
        step,
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        global_grad_norm,
        scalar_float(max_global_grad_norm),
    )


def multi_tensor_lamb_stage2_cuda(
    chunk_size,
    noop_flag,
    tensor_lists,
    per_tensor_param_norm,
    per_tensor_update_norm,
    lr,
    weight_decay,
    use_nvlamb_python=None,
):
    return _ops.amp_multi_tensor_lamb_stage2_cuda(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        per_tensor_param_norm,
        per_tensor_update_norm,
        scalar_float(lr),
        scalar_float(weight_decay),
        use_nvlamb_python,
    )


def multi_tensor_adam(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    return _ops.amp_multi_tensor_adam(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        step,
        mode,
        bias_correction,
        scalar_float(weight_decay),
    )


def multi_tensor_adam_capturable(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
    inv_scale,
):
    return _ops.amp_multi_tensor_adam_capturable(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        lr,
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        step,
        mode,
        bias_correction,
        scalar_float(weight_decay),
        inv_scale,
    )


def multi_tensor_adam_capturable_master(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
    inv_scale,
):
    return _ops.amp_multi_tensor_adam_capturable_master(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        lr,
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        step,
        mode,
        bias_correction,
        scalar_float(weight_decay),
        inv_scale,
    )


def multi_tensor_adagrad(chunk_size, noop_flag, tensor_lists, lr, epsilon, mode, weight_decay):
    return _ops.amp_multi_tensor_adagrad(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        scalar_float(lr),
        scalar_float(epsilon),
        mode,
        scalar_float(weight_decay),
    )


def multi_tensor_novograd(
    chunk_size,
    noop_flag,
    tensor_lists,
    grad_norms,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    bias_correction,
    weight_decay,
    grad_averaging,
    mode,
    norm_type,
):
    return _ops.amp_multi_tensor_novograd(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        grad_norms,
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        step,
        bias_correction,
        scalar_float(weight_decay),
        grad_averaging,
        mode,
        norm_type,
    )


def multi_tensor_lamb(
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
    use_nvlamb_python=None,
):
    return _ops.amp_multi_tensor_lamb(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        scalar_float(lr),
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        step,
        bias_correction,
        scalar_float(weight_decay),
        grad_averaging,
        mode,
        global_grad_norm,
        scalar_float(max_grad_norm),
        use_nvlamb_python,
    )


def multi_tensor_lamb_mp(
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
    use_nvlamb_python,
    found_inf,
    inv_scale,
):
    return _ops.amp_multi_tensor_lamb_mp(
        chunk_size,
        noop_flag,
        tensor_list_arg(tensor_lists),
        lr,
        scalar_float(beta1),
        scalar_float(beta2),
        scalar_float(epsilon),
        step,
        bias_correction,
        scalar_float(weight_decay),
        grad_averaging,
        mode,
        global_grad_norm,
        max_grad_norm,
        use_nvlamb_python,
        found_inf,
        inv_scale,
    )


def update_scale_hysteresis(
    current_scale,
    growth_tracker,
    hysteresis_tracker,
    found_inf,
    growth_factor,
    backoff_factor,
    growth_interval,
    hysteresis,
):
    return _ops.amp_update_scale_hysteresis(
        current_scale,
        growth_tracker,
        hysteresis_tracker,
        found_inf,
        scalar_float(growth_factor),
        scalar_float(backoff_factor),
        growth_interval,
        hysteresis,
    )
