import torch

from apex._custom_ops import load_custom_op_library, scalar_float


load_custom_op_library("_fast_layer_norm", __file__)
_ops = torch.ops.apex


def ln_fwd(x, gamma, beta, epsilon):
    return _ops.fast_layer_norm_ln_fwd(
        x,
        gamma,
        beta,
        scalar_float(epsilon),
    )


def ln_bwd(dz, x_or_z, mu, rsigma, gamma, beta, memory_efficient):
    return _ops.fast_layer_norm_ln_bwd(
        dz,
        x_or_z,
        mu,
        rsigma,
        gamma,
        beta,
        memory_efficient,
    )
