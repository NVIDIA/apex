import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_cudnn_gbn_lib", __file__)
_ops = torch.ops.apex


def _int_list(values):
    return [scalar_int(value) for value in values]


def forward(
    x,
    scale,
    bias,
    running_mean,
    running_var,
    minibatch_mean,
    minibatch_inv_var,
    momentum,
    epsilon,
    bn_group,
    rank_id,
    peer_buffers,
):
    return _ops.cudnn_gbn_forward(
        x,
        scale,
        bias,
        running_mean,
        running_var,
        minibatch_mean,
        minibatch_inv_var,
        scalar_float(momentum),
        scalar_float(epsilon),
        scalar_int(bn_group),
        scalar_int(rank_id),
        _int_list(peer_buffers),
    )


def backward(
    x, dy, scale, minibatch_mean, minibatch_inv_var, epsilon, bn_group, rank_id, peer_buffers
):
    return _ops.cudnn_gbn_backward(
        x,
        dy,
        scale,
        minibatch_mean,
        minibatch_inv_var,
        scalar_float(epsilon),
        scalar_int(bn_group),
        scalar_int(rank_id),
        _int_list(peer_buffers),
    )
