# © 2023 NVIDIA CORPORATION & AFFILIATES

from math import prod

import torch
import triton
from torch.autograd import Function

from apex.contrib.openfold_triton._layer_norm_backward_kernels import (
    PARTIAL_REDUCE_MIN,
    _layer_norm_backward_buf_reduce,
    _layer_norm_backward_dw_db_partial,
    _layer_norm_backward_dw_db_partial_strided,
    _layer_norm_backward_dx,
    _layer_norm_backward_dx_strided,
)
from apex.contrib.openfold_triton._layer_norm_forward_kernels import (
    _layer_norm_forward,
    _layer_norm_forward_strided,
)

# TODO: Find a more elegant approach to cache tuned results.
_M_BUFSIZE_CACHE = dict()


class LayerNormSmallShapeOptImpl(Function):
    @staticmethod
    def forward(ctx, inputs, normalized_shape, weight, bias, eps=1e-05):
        if not inputs.is_contiguous() and normalized_shape != inputs.shape[-1:]:
            raise ValueError(
                f"This implementation only support normalizing along the last dimension for "
                f"noncontiguous inputs. I.e., we expect "
                f"normalized_shape={tuple(inputs.shape[-1:])}, but got {normalized_shape} instead"
            )
        if not inputs.is_contiguous() and inputs.dim() != 4:
            raise ValueError(
                f"This implementation only supports 4-dim noncontiguous inputs, but got "
                f"{inputs.dim()} instead"
            )

        normalized_degree = len(normalized_shape)
        layer_shape = inputs.shape[:-normalized_degree]
        M, N = prod(layer_shape), prod(normalized_shape)

        x_invstd = torch.empty(M, dtype=torch.float32, device=inputs.device)
        x_mean = torch.empty(M, dtype=torch.float32, device=inputs.device)
        y = torch.empty(inputs.shape, dtype=inputs.dtype, device=inputs.device)

        grid = lambda kwargs: (triton.cdiv(kwargs["M"], kwargs["M_BLOCK"]),)
        if inputs.is_contiguous():
            _layer_norm_forward[grid](
                x_ptr=inputs,
                w_ptr=weight,
                b_ptr=bias,
                eps=eps,
                x_invstd_ptr=x_invstd,
                x_mean_ptr=x_mean,
                y_ptr=y,
                M=M,
                N=N,
            )
        else:
            D0, D1, D2, D3 = inputs.shape
            S0, S1, S2, S3 = inputs.stride()
            _layer_norm_forward_strided[grid](
                x_ptr=inputs,
                w_ptr=weight,
                b_ptr=bias,
                eps=eps,
                x_invstd_ptr=x_invstd,
                x_mean_ptr=x_mean,
                y_ptr=y,
                M=M,
                N=N,
                D0=D0,
                D1=D1,
                D2=D2,
                D3=D3,
                S0=S0,
                S1=S1,
                S2=S2,
                S3=S3,
            )

        ctx.save_for_backward(inputs, weight, x_invstd, x_mean)
        ctx.flatten_shape = M, N
        return y

    @staticmethod
    def backward(ctx, d_y):
        inputs, weight, x_invstd, x_mean = ctx.saved_tensors
        M, N = ctx.flatten_shape
        d_inputs = torch.empty_like(inputs)
        d_weight = torch.empty_like(weight)
        d_bias = torch.empty_like(weight)

        # %% Separated kernels, similar to Inductor.
        # 1. dX.
        grid = lambda kwargs: (triton.cdiv(kwargs["M"], kwargs["M_BLOCK"]),)
        if inputs.is_contiguous():
            _layer_norm_backward_dx[grid](
                dy_ptr=d_y,
                x_ptr=inputs,
                w_ptr=weight,
                x_invstd_ptr=x_invstd,
                x_mean_ptr=x_mean,
                dx_ptr=d_inputs,
                M=M,
                N=N,
            )
        else:
            D0, D1, D2, D3 = inputs.shape
            S0, S1, S2, S3 = inputs.stride()
            _layer_norm_backward_dx_strided[grid](
                dy_ptr=d_y,
                x_ptr=inputs,
                w_ptr=weight,
                x_invstd_ptr=x_invstd,
                x_mean_ptr=x_mean,
                dx_ptr=d_inputs,
                M=M,
                N=N,
                D0=D0,
                D1=D1,
                D2=D2,
                D3=D3,
                S0=S0,
                S1=S1,
                S2=S2,
                S3=S3,
            )
        # 2. dW and db.
        key = (M, N, inputs.is_contiguous())
        M_BUFSIZE = _M_BUFSIZE_CACHE.get(key, triton.cdiv(M, PARTIAL_REDUCE_MIN))
        dw_partial_buf = torch.empty(
            [N, M_BUFSIZE], dtype=torch.float32, device=d_y.device
        )
        db_partial_buf = torch.empty(
            [N, M_BUFSIZE], dtype=torch.float32, device=d_y.device
        )
        grid = lambda kwargs: (
            triton.cdiv(M, kwargs["M_PARTIAL_REDUCE"]),
            triton.cdiv(N, kwargs["N_BLOCK"]),
        )
        if inputs.is_contiguous():
            _layer_norm_backward_dw_db_partial[grid](
                dy_ptr=d_y,
                x_ptr=inputs,
                x_invstd_ptr=x_invstd,
                x_mean_ptr=x_mean,
                dw_partial_buf_ptr=dw_partial_buf,
                db_partial_buf_ptr=db_partial_buf,
                M=M,
                N=N,
                BUF_N_STRIDE=M_BUFSIZE,
            )
            M_PARTIAL_REDUCE = _layer_norm_backward_dw_db_partial.best_config.kwargs[
                "M_PARTIAL_REDUCE"
            ]
        else:
            _layer_norm_backward_dw_db_partial_strided[grid](
                dy_ptr=d_y,
                x_ptr=inputs,
                x_invstd_ptr=x_invstd,
                x_mean_ptr=x_mean,
                dw_partial_buf_ptr=dw_partial_buf,
                db_partial_buf_ptr=db_partial_buf,
                M=M,
                N=N,
                BUF_N_STRIDE=M_BUFSIZE,
                D0=D0,
                D1=D1,
                D2=D2,
                D3=D3,
                S0=S0,
                S1=S1,
                S2=S2,
                S3=S3,
            )
            M_PARTIAL_REDUCE = (
                _layer_norm_backward_dw_db_partial_strided.best_config.kwargs[
                    "M_PARTIAL_REDUCE"
                ]
            )
        # 2.1. Reduce partial buffers, which can be overlapped.
        M_BUFSIZE = triton.cdiv(M, M_PARTIAL_REDUCE)
        _M_BUFSIZE_CACHE[key] = M_BUFSIZE
        grid = (triton.next_power_of_2(N),)
        _layer_norm_backward_buf_reduce[grid](
            partial_buf_ptr=dw_partial_buf,
            output_ptr=d_weight,
            N=N,
            M=M_BUFSIZE,
            N_STRIDE=dw_partial_buf.stride(0),
            M_STRIDE=dw_partial_buf.stride(1),
            num_warps=1,
        )
        _layer_norm_backward_buf_reduce[grid](
            partial_buf_ptr=db_partial_buf,
            output_ptr=d_bias,
            N=N,
            M=M_BUFSIZE,
            N_STRIDE=db_partial_buf.stride(0),
            M_STRIDE=db_partial_buf.stride(1),
            num_warps=1,
        )

        return d_inputs, None, d_weight, d_bias, None
