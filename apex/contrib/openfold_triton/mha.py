# © 2023 NVIDIA CORPORATION & AFFILIATES

import math
from typing import Optional

import torch
import triton
from einops import rearrange

from apex.contrib.openfold_triton._mha_kernel import (
    _attention_core,
    _bwd_kernel,
    _bwd_preprocess,
)

# whether TRITON MHA is enabled or not
_TRI_MHA_ENABLED = False


def is_enabled() -> Optional[bool]:
    global _TRI_MHA_ENABLED
    return _TRI_MHA_ENABLED


def enable() -> None:
    global _TRI_MHA_ENABLED
    _TRI_MHA_ENABLED = True


def disable() -> None:
    global _TRI_MHA_ENABLED
    _TRI_MHA_ENABLED = False


# TODO: support q.shape [1, 1024, 8, 256, 8]
def CanSchTriMHA(in_shape, has_bias=True, inf=1e9, training=True):
    if has_bias == False:  # skip bias is None
        return False
    if inf != 1e9:  # skip inf != 1e9
        return False

    lst_3d = in_shape[-3:]
    skip_neg2_dim = in_shape[:3] + in_shape[-1:]
    if not training and (
        in_shape == [1, 538, 4, 538, 16]
        or in_shape == [1, 585, 4, 585, 16]
        or in_shape == [1, 538, 4, 538, 32]
        or in_shape == [1, 585, 4, 585, 32]
        or in_shape == [1, 128, 8, 585, 32]
        or in_shape == [1, 128, 8, 538, 32]
        or lst_3d == [8, 128, 32]
        or skip_neg2_dim == [1, 1024, 8, 8]
        or skip_neg2_dim == [1, 128, 4, 32]
        or skip_neg2_dim == [1, 128, 8, 32]
    ):  # eval
        return False  # skip eval
    if (
        in_shape == [1, 256, 4, 256, 16]
        or in_shape == [1, 128, 4, 256, 16]
        or in_shape == [1, 64, 4, 256, 16]
        or in_shape == [1, 32, 4, 256, 16]
    ):  # 7.26%
        return True
    elif (
        in_shape == [1, 128, 8, 256, 32]
        or in_shape == [1, 64, 8, 256, 32]
        or in_shape == [1, 32, 8, 256, 32]
        or in_shape == [1, 16, 8, 256, 32]
    ):  # 21.77%
        return True
    elif (
        in_shape == [1, 256, 8, 128, 32]
        or in_shape == [1, 128, 8, 128, 32]
        or in_shape == [1, 64, 8, 128, 32]
        or in_shape == [1, 32, 8, 128, 32]
    ):  # 21.77% no bias
        return True
    elif (
        in_shape == [1, 256, 4, 256, 32]
        or in_shape == [1, 128, 4, 256, 32]
        or in_shape == [1, 64, 4, 256, 32]
        or in_shape == [1, 32, 4, 256, 32]
    ):  # 47.17%
        return True
    else:  # not support
        return False


# tune hyper params for each workload
def schedule_triton_mha(in_shape, fwd=True):
    # default
    ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 0]
    if in_shape == [256, 4, 256, 16]:
        ret = [64, 32, 2, 4] if fwd else [64, 64, 4, 0]
    elif in_shape == [128, 4, 256, 16]:
        ret = [64, 32, 2, 4] if fwd else [64, 64, 4, 0]
    elif in_shape == [64, 4, 256, 16]:
        ret = [64, 32, 2, 4] if fwd else [64, 64, 4, 0]
    elif in_shape == [32, 4, 256, 16]:
        ret = [64, 32, 2, 4] if fwd else [64, 64, 4, 0]
    # [*, 8, 256, 32]
    elif in_shape == [128, 8, 256, 32]:  # DAP1
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 1]
    elif in_shape == [64, 8, 256, 32]:  # DAP2
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 1]
    elif in_shape == [32, 8, 256, 32]:  # DAP4
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 1]
    elif in_shape == [16, 8, 256, 32]:  # DAP8
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 1]
    # [*, 8, 128, 32]
    elif in_shape == [256, 8, 128, 32]:  # DAP1
        ret = [64, 64, 4, 3] if fwd else [128, 64, 4, 1]
    elif in_shape == [128, 8, 128, 32]:  # DAP2
        ret = [128, 64, 4, 2] if fwd else [64, 64, 2, 0]
    elif in_shape == [64, 8, 128, 32]:  # DAP4
        ret = [128, 64, 4, 2] if fwd else [64, 64, 2, 0]
    elif in_shape == [32, 8, 128, 32]:  # DAP8
        ret = [128, 64, 4, 2] if fwd else [64, 64, 2, 0]
    # [*, 4, 256, 32]
    elif in_shape == [256, 4, 256, 32]:  # DAP1
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 0]
    elif in_shape == [128, 4, 256, 32]:  # DAP2
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 1]
    elif in_shape == [64, 4, 256, 32]:  # DAP4
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 1]
    elif in_shape == [32, 4, 256, 32]:  # DAP8
        ret = [64, 32, 2, 3] if fwd else [128, 64, 8, 0]
    return ret[0], ret[1], ret[2], ret[3]


class FusedAttenionCoreFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask=None, bias=None, inf=1000000000.0, is_training=True):
        q_ori_size = len(q.size())
        if q_ori_size == 5:
            q = rearrange(q, "1 b2 h n d -> (1 b2) h n d")
            k = rearrange(k, "1 b2 h n d -> (1 b2) h n d")
            v = rearrange(v, "1 b2 h n d -> (1 b2) h n d")
        if bias is not None:
            if len(bias.size()) == 5:
                bias = rearrange(bias, "1 b2 h n d -> (1 b2) h n d")

        if mask is not None and len(mask.size()) == 5:
            mask = rearrange(mask, "1 b 1 1 e -> b 1 1 e")

        batch = 1
        sm_scale = 1.0 / math.sqrt(q.size(-1))
        # q *= sm_scale
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv

        if not is_training:
            Lk = max(triton.next_power_of_2(Lk), 16)

        assert Lk in {16, 32, 64, 128}

        o = torch.empty_like(q)

        Z, H, N_CTX, H_DIM = q.shape
        grid = lambda META: (triton.cdiv(N_CTX, META["BLOCK_M"]), Z * H)
        l = torch.empty(
            (q.shape[-4], q.shape[-3], q.shape[-2]),
            device=q.device,
            dtype=torch.float32,
        )
        m = torch.empty(
            (q.shape[-4], q.shape[-3], q.shape[-2]),
            device=q.device,
            dtype=torch.float32,
        )
        # BLOCK_M, BLOCK_N, num_warps, num_stages  = 64, 64, 2, 3
        BLOCK_M, BLOCK_N, num_warps, num_stages = schedule_triton_mha(
            list(q.shape), fwd=True
        )
        if bias != None:
            bias = bias.expand(Z, H, N_CTX, N_CTX)
        bias_strides = (
            (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
            if bias is not None
            else (0, 0, 0, 0)
        )
        if mask != None:
            mask = mask.expand(-1, q.shape[1], q.shape[2], -1)
        mask_strides = (
            (mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3))
            if mask is not None
            else (0, 0, 0, 0)
        )

        _attention_core[grid](
            q,
            k,
            v,
            mask,
            bias,
            sm_scale,
            l,
            m,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            *bias_strides,
            *mask_strides,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            batch,  # 256 8 128 1
            inf=inf,
            IS_TRAINING=is_training,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            use_mask=(mask != None),
            use_bias=(bias != None),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        o = o.contiguous()
        # print(h.asm["ttgir"])
        if is_training:
            ctx.save_for_backward(q, k, v, o, m, l, bias)
            ctx.grid = grid
            ctx.sm_scale = sm_scale
            ctx.BLOCK_DMODEL = Lk
            ctx.mask = mask
            ctx.inf = inf
        if q_ori_size == 5:
            o = rearrange(o, "a b c d -> 1 a b c d")
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, m, l, bias = ctx.saved_tensors
        ori_do_size = len(do.size())
        if ori_do_size == 5:
            do = rearrange(do, "1 a b c d -> a b c d")
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # bias.dtype
        Z, H, N_CTX, H_DIM = q.shape[-4], q.shape[-3], q.shape[-2], q.shape[-1]
        dp = torch.zeros((Z, H, N_CTX, N_CTX), dtype=torch.float32, device="cuda")

        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        mask = ctx.mask
        inf = ctx.inf

        BLOCK = 128
        BLOCK_HEADDIM = max(triton.next_power_of_2(H_DIM), 16)
        grid = (triton.cdiv(N_CTX, BLOCK) * Z * H, 1)
        _bwd_preprocess[grid](
            o,
            do,
            l,
            do_scaled,
            delta,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            BLOCK_M=BLOCK,
            D_HEAD=BLOCK_HEADDIM,
        )

        if bias is not None:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            assert bias.stride(-1) == 1
            bias = bias.expand(Z, H, N_CTX, N_CTX)

        # if mask is not None:
        #    mask = mask.expand(Z, H, N_CTX, N_CTX)

        bias_strides = (
            (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
            if bias is not None
            else (0, 0, 0, 0)
        )
        mask_strides = (
            (mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3))
            if mask is not None
            else (0, 0, 0, 0)
        )

        # BLOCK_M, BLOCK_N = 128, 64
        BLOCK_M, BLOCK_N, num_warps, num_stages = schedule_triton_mha(
            list(q.shape), fwd=False
        )
        # grid = lambda META: (triton.cdiv(N_CTX, META["BLOCK_N"]), Z * H)
        # grid = lambda META: (Z * H, triton.cdiv(N_CTX, META["BLOCK_N"]))
        # grid = lambda META: (triton.cdiv(N_CTX, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        #            Z * H)
        grid = lambda META: (Z * H,)
        _bwd_kernel[grid](
            q,
            k,
            v,
            mask,
            bias,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            dp,
            l,
            m,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            *mask_strides,
            *bias_strides,
            dp.stride(0),
            dp.stride(1),
            dp.stride(2),
            dp.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            # ctx.grid[0], # to delete
            inf=inf,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            use_mask=(mask != None),
            use_bias=(bias != None),
            num_warps=num_warps,
            num_stages=num_stages,
            SEQUENCE_PARALLEL=False,
        )
        dB = None
        if bias is not None:
            dB = torch.sum(dp, dim=-4, keepdim=True)
            if len(bias.size()) == 4:
                dB = rearrange(dB, "b2 h n d -> 1 b2 h n d")
        # print(h.asm["ttgir"])

        if ori_do_size == 5:
            dq = rearrange(dq, "b2 h n d -> 1 b2 h n d")
            dk = rearrange(dk, "b2 h n d -> 1 b2 h n d")
            dv = rearrange(dv, "b2 h n d -> 1 b2 h n d")

        return dq, dk, dv, None, dB, None, None


AttnTri = FusedAttenionCoreFunc.apply


def _attention_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
) -> torch.Tensor:
    # query:  [*, num_heads, Q, c_hidden]
    # key:    [*, num_heads, K, c_hidden]
    # value:  [*, num_heads, V, c_hidden]
    # mask:   Logit mask tensor broadcastable to [*, num_heads, Q, K]
    # bias:   Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
    # inf:    Safe infinity value.
    # assuming K == V

    key = torch.swapdims(key, -2, -1)
    # key: [*, num_heads, c_hidden, K]

    scaling = 1.0 / math.sqrt(query.size(-1))
    a = torch.matmul(query * scaling, key)
    # a: [*, num_heads, Q, K]

    a += (mask - 1.0) * inf
    # a: [*, num_heads, Q, K]

    a += bias
    # a: [*, num_heads, Q, K]

    a = torch.softmax(a, dim=-1)
    # a: [*, num_heads, Q, K]

    a = torch.matmul(a, value)
    # a: [*, num_heads, Q, c_hidden]

    return a


def _attention_no_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    inf: float,
) -> torch.Tensor:
    # query:  [*, num_heads, Q, c_hidden]
    # key:    [*, num_heads, K, c_hidden]
    # value:  [*, num_heads, V, c_hidden]
    # mask:   Logit mask tensor broadcastable to [*, num_heads, Q, K]
    # bias:   Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
    # inf:    Safe infinity value.
    # assuming K == V

    key = torch.swapdims(key, -2, -1)
    # key: [*, num_heads, c_hidden, K]

    scaling = 1.0 / math.sqrt(query.size(-1))
    a = torch.matmul(query * scaling, key)
    # a: [*, num_heads, Q, K]

    a += (mask - 1.0) * inf
    # a: [*, num_heads, Q, K]

    a = torch.softmax(a, dim=-1)
    # a: [*, num_heads, Q, K]

    a = torch.matmul(a, value)
    # a: [*, num_heads, Q, c_hidden]

    return a


AttnBiasJIT = torch.compile(_attention_bias)
AttnNoBiasJIT = torch.compile(_attention_no_bias)
