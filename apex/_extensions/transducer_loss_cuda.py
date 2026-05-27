import torch

from apex._custom_ops import load_custom_op_library, scalar_int


load_custom_op_library("_transducer_loss_cuda", __file__)
_ops = torch.ops.apex


def forward(x, label, f_len, y_len, batch_offset, max_f_len, blank_idx, opt, packed_input):
    return _ops.transducer_loss_forward(
        x,
        label,
        f_len,
        y_len,
        batch_offset,
        scalar_int(max_f_len),
        scalar_int(blank_idx),
        scalar_int(opt),
        packed_input,
    )


def backward(
    x,
    loss_grad,
    alpha,
    beta,
    f_len,
    y_len,
    label,
    batch_offset,
    max_f_len,
    blank_idx,
    opt,
    fuse_softmax_backward,
    packed_input,
):
    return _ops.transducer_loss_backward(
        x,
        loss_grad,
        alpha,
        beta,
        f_len,
        y_len,
        label,
        batch_offset,
        scalar_int(max_f_len),
        scalar_int(blank_idx),
        scalar_int(opt),
        fuse_softmax_backward,
        packed_input,
    )
