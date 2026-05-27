import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_fmhalib", __file__)
_ops = torch.ops.apex


def fwd(qkv, cu_seqlens, p_dropout, max_seq_len, is_training, is_nl, zero_tensors, gen):
    return _ops.fmha_fwd(
        qkv,
        cu_seqlens,
        scalar_float(p_dropout),
        scalar_int(max_seq_len),
        bool(is_training),
        bool(is_nl),
        bool(zero_tensors),
        gen,
    )


def fwd_nl(qkv, cu_seqlens, p_dropout, max_seq_len, is_training, is_nl, zero_tensors, gen):
    return fwd(qkv, cu_seqlens, p_dropout, max_seq_len, is_training, True, zero_tensors, gen)


def bwd(dout, qkv, softmax, cu_seqlens, p_dropout, max_seq_len, zero_tensors):
    return _ops.fmha_bwd(
        dout,
        qkv,
        softmax,
        cu_seqlens,
        scalar_float(p_dropout),
        scalar_int(max_seq_len),
        bool(zero_tensors),
    )


def bwd_nl(dout, qkv, softmax, cu_seqlens, p_dropout, max_seq_len, zero_tensors):
    return _ops.fmha_bwd_nl(
        dout,
        qkv,
        softmax,
        cu_seqlens,
        scalar_float(p_dropout),
        scalar_int(max_seq_len),
        bool(zero_tensors),
    )
