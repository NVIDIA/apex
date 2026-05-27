import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_transducer_joint_cuda", __file__)
_ops = torch.ops.apex


def forward(
    f,
    g,
    f_len,
    g_len,
    batch_offset,
    packed_batch,
    opt,
    pack_output,
    relu,
    dropout,
    dropout_prob,
    tile_size,
):
    return _ops.transducer_joint_forward(
        f,
        g,
        f_len,
        g_len,
        batch_offset,
        scalar_int(packed_batch),
        scalar_int(opt),
        pack_output,
        relu,
        dropout,
        scalar_float(dropout_prob),
        scalar_int(tile_size),
    )


def backward(input, f_len, g_len, batch_offset, max_f_len, max_g_len, pack_output, scale):
    return _ops.transducer_joint_backward(
        list(input),
        f_len,
        g_len,
        batch_offset,
        scalar_int(max_f_len),
        scalar_int(max_g_len),
        pack_output,
        scalar_float(scale),
    )
