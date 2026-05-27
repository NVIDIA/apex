import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_bnp", __file__)
_ops = torch.ops.apex


def _optional_ptr(value):
    return None if value is None else scalar_int(value)


def get_buffer_size(bn_sync_steps):
    return _ops.bnp_get_buffer_size(scalar_int(bn_sync_steps))


def get_data_ptr(data):
    return _ops.bnp_get_data_ptr(data)


def get_remote_data_ptr(handle, offset):
    return _ops.bnp_get_remote_data_ptr(handle, scalar_int(offset))


def close_remote_data(handle):
    return _ops.bnp_close_remote_data(handle)


def bn_fwd_nhwc(
    x,
    scale,
    bias,
    running_mean,
    running_inv_var,
    minibatch_mean,
    minibatch_inv_var,
    ret_cta,
    momentum,
    epsilon,
    fuse_relu,
    my_data,
    pair_data,
    pair_data2,
    pair_data3,
    bn_group,
    magic,
    occupancy,
    grid_dim_x,
    coop,
):
    return _ops.bnp_bn_fwd_nhwc(
        x,
        scale,
        bias,
        running_mean,
        running_inv_var,
        minibatch_mean,
        minibatch_inv_var,
        ret_cta,
        scalar_float(momentum),
        scalar_float(epsilon),
        bool(fuse_relu),
        _optional_ptr(my_data),
        _optional_ptr(pair_data),
        _optional_ptr(pair_data2),
        _optional_ptr(pair_data3),
        scalar_int(bn_group),
        magic,
        scalar_int(occupancy),
        scalar_int(grid_dim_x),
        bool(coop),
    )


def bn_fwd_eval_nhwc(x, scale, bias, running_mean, running_inv_var, ret_cta, bn_group, momentum, epsilon, fuse_relu):
    return _ops.bnp_bn_fwd_eval_nhwc(
        x,
        scale,
        bias,
        running_mean,
        running_inv_var,
        ret_cta,
        scalar_int(bn_group),
        scalar_float(momentum),
        scalar_float(epsilon),
        bool(fuse_relu),
    )


def bn_bwd_nhwc(
    x,
    dy,
    scale,
    bias,
    running_mean,
    running_inv_var,
    minibatch_mean,
    minibatch_inv_var,
    ret_cta,
    momentum,
    epsilon,
    fuse_relu,
    my_data,
    pair_data,
    pair_data2,
    pair_data3,
    bn_group,
    magic,
    occupancy,
    grid_dim_x,
    coop,
):
    return _ops.bnp_bn_bwd_nhwc(
        x,
        dy,
        scale,
        bias,
        running_mean,
        running_inv_var,
        minibatch_mean,
        minibatch_inv_var,
        ret_cta,
        scalar_float(momentum),
        scalar_float(epsilon),
        bool(fuse_relu),
        _optional_ptr(my_data),
        _optional_ptr(pair_data),
        _optional_ptr(pair_data2),
        _optional_ptr(pair_data3),
        scalar_int(bn_group),
        magic,
        scalar_int(occupancy),
        scalar_int(grid_dim_x),
        bool(coop),
    )


def bn_fwd_nhwc_occupancy():
    return _ops.bnp_bn_fwd_nhwc_occupancy()


def bn_bwd_nhwc_occupancy():
    return _ops.bnp_bn_bwd_nhwc_occupancy()


def bn_addrelu_fwd_nhwc(
    x,
    z,
    scale,
    bias,
    running_mean,
    running_inv_var,
    minibatch_mean,
    minibatch_inv_var,
    bitmask,
    ret_cta,
    momentum,
    epsilon,
    my_data,
    pair_data,
    pair_data2,
    pair_data3,
    bn_group,
    magic,
    occupancy,
    grid_dim_x,
    coop,
):
    return _ops.bnp_bn_addrelu_fwd_nhwc(
        x,
        z,
        scale,
        bias,
        running_mean,
        running_inv_var,
        minibatch_mean,
        minibatch_inv_var,
        bitmask,
        ret_cta,
        scalar_float(momentum),
        scalar_float(epsilon),
        _optional_ptr(my_data),
        _optional_ptr(pair_data),
        _optional_ptr(pair_data2),
        _optional_ptr(pair_data3),
        scalar_int(bn_group),
        magic,
        scalar_int(occupancy),
        scalar_int(grid_dim_x),
        bool(coop),
    )


def bn_addrelu_fwd_eval_nhwc(x, z, scale, bias, running_mean, running_inv_var, ret_cta, bn_group, momentum, epsilon):
    return _ops.bnp_bn_addrelu_fwd_eval_nhwc(
        x,
        z,
        scale,
        bias,
        running_mean,
        running_inv_var,
        ret_cta,
        scalar_int(bn_group),
        scalar_float(momentum),
        scalar_float(epsilon),
    )


def bn_addrelu_bwd_nhwc(
    x,
    dy,
    scale,
    bias,
    running_mean,
    running_inv_var,
    minibatch_mean,
    minibatch_inv_var,
    bitmask,
    ret_cta,
    momentum,
    epsilon,
    my_data,
    pair_data,
    pair_data2,
    pair_data3,
    bn_group,
    magic,
    occupancy,
    grid_dim_x,
    coop,
):
    return _ops.bnp_bn_addrelu_bwd_nhwc(
        x,
        dy,
        scale,
        bias,
        running_mean,
        running_inv_var,
        minibatch_mean,
        minibatch_inv_var,
        bitmask,
        ret_cta,
        scalar_float(momentum),
        scalar_float(epsilon),
        _optional_ptr(my_data),
        _optional_ptr(pair_data),
        _optional_ptr(pair_data2),
        _optional_ptr(pair_data3),
        scalar_int(bn_group),
        magic,
        scalar_int(occupancy),
        scalar_int(grid_dim_x),
        bool(coop),
    )


def bn_addrelu_fwd_nhwc_occupancy():
    return _ops.bnp_bn_addrelu_fwd_nhwc_occupancy()


def bn_addrelu_bwd_nhwc_occupancy():
    return _ops.bnp_bn_addrelu_bwd_nhwc_occupancy()
