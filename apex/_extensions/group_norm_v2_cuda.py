import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_group_norm_v2_cuda", __file__)


def gn(x, w, b, eps, silu, num_groups, mean_var_out=None, sm_margin=0):
    return torch.ops.apex.group_norm_v2_gn(
        x,
        w,
        b,
        scalar_float(eps),
        bool(silu),
        scalar_int(num_groups),
        mean_var_out,
        scalar_int(sm_margin),
    )


def gn_bwd(grad_output, x, w, b, mean_var, eps, silu, num_groups, sm_margin=0):
    return torch.ops.apex.group_norm_v2_gn_bwd(
        grad_output,
        x,
        w,
        b,
        mean_var,
        scalar_float(eps),
        bool(silu),
        scalar_int(num_groups),
        scalar_int(sm_margin),
    )
