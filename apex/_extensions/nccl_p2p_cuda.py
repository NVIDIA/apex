import torch

from apex._custom_ops import load_custom_op_library, scalar_int


load_custom_op_library("_nccl_p2p_cuda", __file__)
_ops = torch.ops.apex


def get_unique_nccl_id(n):
    return _ops.nccl_p2p_get_unique_nccl_id(scalar_int(n))


def init_nccl_comm(unique_nccl_id, my_rank, num_ranks):
    return _ops.nccl_p2p_init_nccl_comm(
        unique_nccl_id,
        scalar_int(my_rank),
        scalar_int(num_ranks),
    )


def left_right_halo_exchange_inplace(
    handle,
    left_rank,
    right_rank,
    left_output_halo,
    right_output_halo,
    left_input_halo,
    right_input_halo,
):
    return _ops.nccl_p2p_left_right_halo_exchange_inplace(
        scalar_int(handle),
        scalar_int(left_rank),
        scalar_int(right_rank),
        left_output_halo,
        right_output_halo,
        left_input_halo,
        right_input_halo,
    )


def left_right_halo_exchange(handle, left_rank, right_rank, left_output_halo, right_output_halo):
    return _ops.nccl_p2p_left_right_halo_exchange(
        scalar_int(handle),
        scalar_int(left_rank),
        scalar_int(right_rank),
        left_output_halo,
        right_output_halo,
    )


def add_delay(delay):
    return _ops.nccl_p2p_add_delay(scalar_int(delay))
