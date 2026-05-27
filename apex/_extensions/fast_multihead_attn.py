import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_fast_multihead_attn", __file__)
_ops = torch.ops.apex


def additive_mask_softmax_dropout_forward(use_mask, is_training, heads, input, pad_mask, dropout_prob):
    return _ops.fast_multihead_attn_additive_mask_softmax_dropout_forward(
        bool(use_mask), bool(is_training), scalar_int(heads), input, pad_mask, scalar_float(dropout_prob)
    )


def additive_mask_softmax_dropout_backward(use_mask, heads, output_grads, softmax_results, dropout_mask, dropout_prob):
    return _ops.fast_multihead_attn_additive_mask_softmax_dropout_backward(
        bool(use_mask), scalar_int(heads), output_grads, softmax_results, dropout_mask, scalar_float(dropout_prob)
    )


def mask_softmax_dropout_forward(use_mask, is_training, heads, input, pad_mask, dropout_prob):
    return _ops.fast_multihead_attn_mask_softmax_dropout_forward(
        bool(use_mask), bool(is_training), scalar_int(heads), input, pad_mask, scalar_float(dropout_prob)
    )


def mask_softmax_dropout_backward(use_mask, heads, output_grads, softmax_results, dropout_mask, padding_mask, dropout_prob):
    return _ops.fast_multihead_attn_mask_softmax_dropout_backward(
        bool(use_mask),
        scalar_int(heads),
        output_grads,
        softmax_results,
        dropout_mask,
        padding_mask,
        scalar_float(dropout_prob),
    )


def encdec_multihead_attn_forward(
    use_mask,
    use_time_mask,
    is_training,
    heads,
    inputs_q,
    inputs_kv,
    input_weights_q,
    input_weights_kv,
    output_weights,
    pad_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_encdec_multihead_attn_forward(
        bool(use_mask),
        bool(use_time_mask),
        bool(is_training),
        scalar_int(heads),
        inputs_q,
        inputs_kv,
        input_weights_q,
        input_weights_kv,
        output_weights,
        pad_mask,
        scalar_float(dropout_prob),
    )


def encdec_multihead_attn_backward(
    heads,
    output_grads,
    matmul2_results,
    dropout_results,
    softmax_results,
    input_lin_q_results,
    input_lin_kv_results,
    inputs_q,
    inputs_kv,
    input_weights_q,
    input_weights_kv,
    output_weights,
    dropout_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_encdec_multihead_attn_backward(
        scalar_int(heads),
        output_grads,
        matmul2_results,
        dropout_results,
        softmax_results,
        input_lin_q_results,
        input_lin_kv_results,
        inputs_q,
        inputs_kv,
        input_weights_q,
        input_weights_kv,
        output_weights,
        dropout_mask,
        scalar_float(dropout_prob),
    )


def encdec_multihead_attn_norm_add_forward(
    use_mask,
    use_time_mask,
    is_training,
    heads,
    inputs_q,
    inputs_kv,
    lyr_nrm_gamma_weights,
    lyr_nrm_beta_weights,
    input_weights_q,
    input_weights_kv,
    output_weights,
    pad_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_encdec_multihead_attn_norm_add_forward(
        bool(use_mask),
        bool(use_time_mask),
        bool(is_training),
        scalar_int(heads),
        inputs_q,
        inputs_kv,
        lyr_nrm_gamma_weights,
        lyr_nrm_beta_weights,
        input_weights_q,
        input_weights_kv,
        output_weights,
        pad_mask,
        scalar_float(dropout_prob),
    )


def encdec_multihead_attn_norm_add_backward(
    heads,
    output_grads,
    matmul2_results,
    dropout_results,
    softmax_results,
    input_lin_q_results,
    input_lin_kv_results,
    lyr_nrm_results,
    lyr_nrm_mean,
    lyr_nrm_invvar,
    inputs_q,
    inputs_kv,
    lyr_nrm_gamma_weights,
    lyr_nrm_beta_weights,
    input_weights_q,
    input_weights_kv,
    output_weights,
    dropout_mask,
    dropout_add_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_encdec_multihead_attn_norm_add_backward(
        scalar_int(heads),
        output_grads,
        matmul2_results,
        dropout_results,
        softmax_results,
        input_lin_q_results,
        input_lin_kv_results,
        lyr_nrm_results,
        lyr_nrm_mean,
        lyr_nrm_invvar,
        inputs_q,
        inputs_kv,
        lyr_nrm_gamma_weights,
        lyr_nrm_beta_weights,
        input_weights_q,
        input_weights_kv,
        output_weights,
        dropout_mask,
        dropout_add_mask,
        scalar_float(dropout_prob),
    )


def self_attn_forward(
    use_mask,
    use_time_mask,
    is_training,
    heads,
    inputs,
    input_weights,
    output_weights,
    pad_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_forward(
        bool(use_mask),
        bool(use_time_mask),
        bool(is_training),
        scalar_int(heads),
        inputs,
        input_weights,
        output_weights,
        pad_mask,
        scalar_float(dropout_prob),
    )


def self_attn_backward(
    heads,
    output_grads,
    matmul2_results,
    dropout_results,
    softmax_results,
    input_lin_results,
    inputs,
    input_weights,
    output_weights,
    dropout_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_backward(
        scalar_int(heads),
        output_grads,
        matmul2_results,
        dropout_results,
        softmax_results,
        input_lin_results,
        inputs,
        input_weights,
        output_weights,
        dropout_mask,
        scalar_float(dropout_prob),
    )


def self_attn_bias_forward(
    use_mask,
    use_time_mask,
    is_training,
    heads,
    inputs,
    input_weights,
    output_weights,
    input_biases,
    output_biases,
    pad_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_bias_forward(
        bool(use_mask),
        bool(use_time_mask),
        bool(is_training),
        scalar_int(heads),
        inputs,
        input_weights,
        output_weights,
        input_biases,
        output_biases,
        pad_mask,
        scalar_float(dropout_prob),
    )


def self_attn_bias_backward(
    heads,
    output_grads,
    matmul2_results,
    dropout_results,
    softmax_results,
    input_lin_results,
    inputs,
    input_weights,
    output_weights,
    dropout_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_bias_backward(
        scalar_int(heads),
        output_grads,
        matmul2_results,
        dropout_results,
        softmax_results,
        input_lin_results,
        inputs,
        input_weights,
        output_weights,
        dropout_mask,
        scalar_float(dropout_prob),
    )


def self_attn_bias_additive_mask_forward(
    use_mask,
    use_time_mask,
    is_training,
    heads,
    inputs,
    input_weights,
    output_weights,
    input_biases,
    output_biases,
    pad_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_bias_additive_mask_forward(
        bool(use_mask),
        bool(use_time_mask),
        bool(is_training),
        scalar_int(heads),
        inputs,
        input_weights,
        output_weights,
        input_biases,
        output_biases,
        pad_mask,
        scalar_float(dropout_prob),
    )


def self_attn_bias_additive_mask_backward(
    heads,
    output_grads,
    matmul2_results,
    dropout_results,
    bmm1_results,
    pad_mask,
    input_lin_results,
    inputs,
    input_weights,
    output_weights,
    dropout_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_bias_additive_mask_backward(
        scalar_int(heads),
        output_grads,
        matmul2_results,
        dropout_results,
        bmm1_results,
        pad_mask,
        input_lin_results,
        inputs,
        input_weights,
        output_weights,
        dropout_mask,
        scalar_float(dropout_prob),
    )


def self_attn_norm_add_forward(
    use_mask,
    use_time_mask,
    is_training,
    heads,
    inputs,
    lyr_nrm_gamma_weights,
    lyr_nrm_beta_weights,
    input_weights,
    output_weights,
    pad_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_norm_add_forward(
        bool(use_mask),
        bool(use_time_mask),
        bool(is_training),
        scalar_int(heads),
        inputs,
        lyr_nrm_gamma_weights,
        lyr_nrm_beta_weights,
        input_weights,
        output_weights,
        pad_mask,
        scalar_float(dropout_prob),
    )


def self_attn_norm_add_backward(
    heads,
    output_grads,
    matmul2_results,
    dropout_results,
    softmax_results,
    input_lin_results,
    lyr_nrm_results,
    lyr_nrm_mean,
    lyr_nrm_invvar,
    inputs,
    lyr_nrm_gamma_weights,
    lyr_nrm_beta_weights,
    input_weights,
    output_weights,
    dropout_mask,
    dropout_add_mask,
    dropout_prob,
):
    return _ops.fast_multihead_attn_self_attn_norm_add_backward(
        scalar_int(heads),
        output_grads,
        matmul2_results,
        dropout_results,
        softmax_results,
        input_lin_results,
        lyr_nrm_results,
        lyr_nrm_mean,
        lyr_nrm_invvar,
        inputs,
        lyr_nrm_gamma_weights,
        lyr_nrm_beta_weights,
        input_weights,
        output_weights,
        dropout_mask,
        dropout_add_mask,
        scalar_float(dropout_prob),
    )
