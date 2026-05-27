import torch

from apex._custom_ops import load_custom_op_library, scalar_int


load_custom_op_library("_peer_memory_cuda", __file__)
_ops = torch.ops.apex


def _shape_arg(shape):
    return [scalar_int(dim) for dim in shape]


def allocate_raw(size):
    return _ops.peer_memory_allocate_raw(scalar_int(size))


def free_raw(raw):
    return _ops.peer_memory_free_raw(scalar_int(raw))


def zero(raw, size):
    return _ops.peer_memory_zero(scalar_int(raw), scalar_int(size))


def get_raw_ipc_address(raw):
    return _ops.peer_memory_get_raw_ipc_address(scalar_int(raw))


def get_raw_peers(ipc_addresses, peer_rank, raw):
    return _ops.peer_memory_get_raw_peers(
        ipc_addresses,
        scalar_int(peer_rank),
        scalar_int(raw),
    )


def blob_view_half(raw, shape, channels_last):
    return _ops.peer_memory_blob_view_half(
        scalar_int(raw),
        _shape_arg(shape),
        bool(channels_last),
    )


def blob_view_float(raw, shape, channels_last):
    return _ops.peer_memory_blob_view_float(
        scalar_int(raw),
        _shape_arg(shape),
        bool(channels_last),
    )


def blob_view_int(raw, shape, channels_last):
    return _ops.peer_memory_blob_view_int(
        scalar_int(raw),
        _shape_arg(shape),
        bool(channels_last),
    )


def push_pull_halos_1d(
    diagnostics,
    explicit_nhwc,
    numSM,
    rank,
    top_zero,
    top_in_halo,
    top_in_transfer,
    top_out_transfer,
    top_out_halo,
    btm_zero,
    btm_in_halo,
    btm_in_transfer,
    btm_out_transfer,
    btm_out_halo,
):
    return _ops.peer_memory_push_pull_halos_1d(
        bool(diagnostics),
        bool(explicit_nhwc),
        scalar_int(numSM),
        scalar_int(rank),
        bool(top_zero),
        top_in_halo,
        top_in_transfer,
        top_out_transfer,
        top_out_halo,
        bool(btm_zero),
        btm_in_halo,
        btm_in_transfer,
        btm_out_transfer,
        btm_out_halo,
    )
