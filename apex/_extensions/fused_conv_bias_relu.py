import torch

from apex._custom_ops import load_custom_op_library, scalar_int


load_custom_op_library("_fused_conv_bias_relu", __file__)
_ops = torch.ops.apex


def _tensor_list(inputs):
    return list(inputs)


def forward(inputs, padding, stride):
    return _ops.fused_conv_bias_relu_forward(
        _tensor_list(inputs),
        scalar_int(padding),
        scalar_int(stride),
    )


def backward(inputs, padding, stride):
    return _ops.fused_conv_bias_relu_backward(
        _tensor_list(inputs),
        scalar_int(padding),
        scalar_int(stride),
    )


def forward_no_relu(inputs, padding, stride):
    return _ops.fused_conv_bias_relu_forward_no_relu(
        _tensor_list(inputs),
        scalar_int(padding),
        scalar_int(stride),
    )


def backward_no_relu(inputs, padding, stride):
    return _ops.fused_conv_bias_relu_backward_no_relu(
        _tensor_list(inputs),
        scalar_int(padding),
        scalar_int(stride),
    )


def forward_mask(inputs, padding, stride):
    return _ops.fused_conv_bias_relu_forward_mask(
        _tensor_list(inputs),
        scalar_int(padding),
        scalar_int(stride),
    )


def forward_cscale_cbias_relu(inputs, padding, stride):
    return _ops.fused_conv_bias_relu_forward_cscale_cbias_relu(
        _tensor_list(inputs),
        scalar_int(padding),
        scalar_int(stride),
    )


def backward_cscale_cbias_relu(inputs, padding, stride):
    return _ops.fused_conv_bias_relu_backward_cscale_cbias_relu(
        _tensor_list(inputs),
        scalar_int(padding),
        scalar_int(stride),
    )
