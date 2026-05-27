import torch

from apex._custom_ops import load_custom_op_library, scalar_int


load_custom_op_library("_fast_bottleneck", __file__)
_ops = torch.ops.apex


def _tensor_list(values):
    return list(values)


def forward(explicit_nhwc, stride_1x1, inputs):
    return _ops.fast_bottleneck_forward(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs)
    )


def backward(explicit_nhwc, stride_1x1, inputs):
    return _ops.fast_bottleneck_backward(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs)
    )


def forward_init(explicit_nhwc, stride_1x1, inputs):
    return _ops.fast_bottleneck_forward_init(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs)
    )


def forward_out1(explicit_nhwc, stride_1x1, inputs, outputs):
    return _ops.fast_bottleneck_forward_out1(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs), _tensor_list(outputs)
    )


def forward_out2(explicit_nhwc, stride_1x1, inputs, outputs):
    return _ops.fast_bottleneck_forward_out2(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs), _tensor_list(outputs)
    )


def forward_out2_mask(explicit_nhwc, stride_1x1, inputs, outputs, thresholdTop, thresholdBottom):
    return _ops.fast_bottleneck_forward_out2_mask(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        thresholdTop,
        thresholdBottom,
    )


def forward_out2_halo(explicit_nhwc, fat_halo_y1, inputs):
    return _ops.fast_bottleneck_forward_out2_halo(
        bool(explicit_nhwc), fat_halo_y1, _tensor_list(inputs)
    )


def forward_out2_halo_corr(explicit_nhwc, slim_halo_y1, inputs, w1by3, out2_part_halo):
    return _ops.fast_bottleneck_forward_out2_halo_corr(
        bool(explicit_nhwc), slim_halo_y1, _tensor_list(inputs), w1by3, out2_part_halo
    )


def forward_out2_pad(explicit_nhwc, stride_1x1, inputs, outputs, out1_pad):
    return _ops.fast_bottleneck_forward_out2_pad(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        out1_pad,
    )


def forward_rest(explicit_nhwc, stride_1x1, inputs, outputs):
    return _ops.fast_bottleneck_forward_rest(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs), _tensor_list(outputs)
    )


def backward_init(explicit_nhwc, stride_1x1, inputs):
    return _ops.fast_bottleneck_backward_init(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs)
    )


def backward_grad_out2(explicit_nhwc, stride_1x1, inputs, outputs):
    return _ops.fast_bottleneck_backward_grad_out2(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs), _tensor_list(outputs)
    )


def backward_grad_out1(explicit_nhwc, stride_1x1, inputs, outputs, grad_out2):
    return _ops.fast_bottleneck_backward_grad_out1(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        grad_out2,
    )


def backward_grad_out1_mask(
    explicit_nhwc, stride_1x1, inputs, outputs, grad_out2, thresholdTop, thresholdBottom
):
    return _ops.fast_bottleneck_backward_grad_out1_mask(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        grad_out2,
        thresholdTop,
        thresholdBottom,
    )


def backward_grad_out1_halo(explicit_nhwc, stride_1x1, inputs, outputs, grad_out2_halo, relu1_halo):
    return _ops.fast_bottleneck_backward_grad_out1_halo(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        grad_out2_halo,
        relu1_halo,
    )


def backward_grad_out1_halo_corr(
    explicit_nhwc, stride_1x1, inputs, w1by3, outputs, grad_out2_halo, relu1_halo, part_grad_out1
):
    return _ops.fast_bottleneck_backward_grad_out1_halo_corr(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        w1by3,
        _tensor_list(outputs),
        grad_out2_halo,
        relu1_halo,
        part_grad_out1,
    )


def backward_wgrad2_pad(explicit_nhwc, stride_1x1, inputs, outputs, input, grad_out2):
    return _ops.fast_bottleneck_backward_wgrad2_pad(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        input,
        grad_out2,
    )


def backward_wgrad2(explicit_nhwc, stride_1x1, inputs, outputs, grad_out2):
    return _ops.fast_bottleneck_backward_wgrad2(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        grad_out2,
    )


def backward_wgrad2_halo(explicit_nhwc, stride_1x1, inputs, outputs, input, grad_out2_halo):
    return _ops.fast_bottleneck_backward_wgrad2_halo(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        input,
        grad_out2_halo,
    )


def backward_wgrad3(explicit_nhwc, stride_1x1, inputs, outputs):
    return _ops.fast_bottleneck_backward_wgrad3(
        bool(explicit_nhwc), scalar_int(stride_1x1), _tensor_list(inputs), _tensor_list(outputs)
    )


def backward_wgrad1(explicit_nhwc, stride_1x1, inputs, outputs, grad_out1):
    return _ops.fast_bottleneck_backward_wgrad1(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        grad_out1,
    )


def backward_rest(explicit_nhwc, stride_1x1, inputs, outputs, grad_out2, grad_out1):
    return _ops.fast_bottleneck_backward_rest(
        bool(explicit_nhwc),
        scalar_int(stride_1x1),
        _tensor_list(inputs),
        _tensor_list(outputs),
        grad_out2,
        grad_out1,
    )
