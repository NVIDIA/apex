# © 2023 NVIDIA CORPORATION & AFFILIATES

import triton
import triton.language as tl


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_fwd():
    configs = []
    for num_stages in [0, 1, 2, 3, 4]:
        for block_m in [32, 64, 128]:
            for block_n in [16, 32, 64, 128]:
                if block_n > block_m:
                    continue
                for num_warps in [1, 2, 4, 8]:
                    if 32 * num_warps * 32 > block_m * block_n:
                        continue
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


"""
@triton.autotune(
    configs=get_configs_fwd(), 
    key=['Z', 'H', 'N_CTX', 'H_DIM', 'IS_TRAINING'],
)
"""


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N_CTX"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["H_DIM"] == args["BLOCK_DMODEL"],
    }
)
@triton.jit
def _attention_core(
    Q,
    K,
    V,
    Mask,
    Bias,
    sm_scale,
    L,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_mz,
    stride_mh,
    stride_mm,
    stride_mn,
    Z,
    H,
    N_CTX,
    H_DIM,
    BATCH,  # 256 8 128 32 1
    inf: tl.constexpr,
    IS_TRAINING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    use_mask: tl.constexpr,
    use_bias: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (
        off_b * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    off_k = (
        off_b * stride_kz
        + off_h * stride_kh
        + offs_n[None, :] * stride_kn
        + offs_d[:, None] * stride_kk
    )
    off_v = (
        off_b * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vk
        + offs_d[None, :] * stride_vn
    )
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointers to bias, mask
    if use_bias:
        batch_2 = Z // BATCH
        off_hz_bias = (off_hz // (batch_2 * H) * H) + (off_hz % H)
        offs_base_bias = (
            off_hz_bias * (N_CTX * N_CTX) + offs_m[:, None] * N_CTX + offs_n[None, :]
        )
        """
        off_b = off_hz // H
        off_h = off_hz % H
        bias_ptrs = Bias + off_b * stride_bz + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
        """

    if use_mask:
        # off_hz_mask = (off_hz // H)
        # offs_base_mask = off_hz_mask * N_CTX
        off_b = off_hz // H
        off_h = off_hz % H
        mask_ptrs = (
            Mask
            + off_b * stride_mz
            + off_h * stride_mh
            + (offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn)
        )

    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < H_DIM, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < H_DIM),
                other=0.0,
            )

    # loop over k, v and update accumulator
    #  (start_m + 1) * BLOCK_M
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if (
            EVEN_N & EVEN_M
        ):  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs)
            else:
                k = tl.load(k_ptrs, mask=offs_d[:, None] < H_DIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs, mask=(start_n + offs_n)[None, :] < N_CTX, other=0.0)
            else:
                k = tl.load(
                    k_ptrs,
                    mask=((start_n + offs_n)[None, :] < N_CTX)
                    & (offs_d[:, None] < H_DIM),
                    other=0.0,
                )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if use_bias:
            qk += tl.dot(q * sm_scale.to(tl.bfloat16), k).to(tl.bfloat16)
            qk += tl.where((start_n + offs_n)[None, :] < N_CTX, 0, -inf).to(tl.bfloat16)
            if EVEN_M & EVEN_N:
                bias_data = tl.load(Bias + offs_base_bias + start_n)
            else:
                bias_load_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                bias_load_mask = tl.where(offs_m[:, None] >= N_CTX, 1.0, bias_load_mask)
                bias_load_mask = tl.where(
                    (start_n + offs_n)[None, :] >= N_CTX, 1.0, bias_load_mask
                )
                bias_data = tl.load(
                    Bias + offs_base_bias + start_n,
                    mask=(bias_load_mask == 0.0),
                    other=0.0,
                )
            qk = qk + bias_data
        else:
            qk += tl.dot(q, k)
            qk += tl.where((start_n + offs_n)[None, :] < N_CTX, 0, -inf)

        qk = qk.to(tl.bfloat16)

        if use_mask:
            if EVEN_M & EVEN_N:
                mask_data = tl.load(mask_ptrs + start_n).to(tl.int32)
            else:
                mask_data = tl.load(
                    mask_ptrs + start_n,
                    mask=(offs_m[:, None] < N_CTX)
                    & ((start_n + offs_n)[None, :] < N_CTX),
                    other=0,
                ).to(tl.int32)
            qk += tl.where(mask_data == 0, -inf, 0.0)

        if use_bias:
            # compute new m
            m_curr = tl.maximum(tl.max(qk, 1), m_prev)
            # correct old l
            l_prev *= tl.exp(m_prev - m_curr)
            # attention weights
            p = tl.exp(qk - m_curr[:, None])
        else:
            m_curr = tl.maximum(tl.max(qk, 1) * sm_scale, m_prev)
            l_prev *= tl.exp(m_prev - m_curr)
            p = tl.exp(qk * sm_scale - m_curr[:, None])

        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1.0 / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)

        if (
            EVEN_N & EVEN_M
        ):  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs)
            else:
                v = tl.load(v_ptrs, mask=offs_d[None, :] < H_DIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
            else:
                v = tl.load(
                    v_ptrs,
                    mask=((start_n + offs_n)[:, None] < N_CTX)
                    & (offs_d[None, :] < H_DIM),
                    other=0.0,
                )
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    if IS_TRAINING:
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, l_prev)
        tl.store(m_ptrs, m_prev)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = (
        off_b * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    out_ptrs = Out + off_o
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc.to(Q.dtype.element_ty))
        else:
            tl.store(out_ptrs, acc.to(Q.dtype.element_ty), mask=offs_n[None, :] < H_DIM)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc.to(Q.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
        else:
            tl.store(
                out_ptrs,
                acc.to(Q.dtype.element_ty),
                mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < H_DIM),
            )
    # tl.store(out_ptrs, acc.to(Q.dtype.element_ty), mask=out_store_mask)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    L,
    NewDO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dok,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


def get_configs_bwd():
    configs = []
    for num_stages in [0, 1, 2, 3, 4]:
        for block_m in [32, 64, 128]:
            for block_n in [16, 32, 64, 128]:
                if block_n > block_m:
                    continue
                for num_warps in [1, 2, 4, 8]:
                    if 32 * num_warps * 32 > block_m * block_n:
                        continue
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=num_stages,
                            num_warps=num_warps,
                            pre_hook=init_to_zero("DQ"),
                        )
                    )
    return configs


"""
@triton.autotune(
    configs=get_configs_bwd(),
    key=['Z', 'H', 'N_CTX', 'H_DIM'],
)
"""


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N_CTX"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["H_DIM"] == args["BLOCK_DMODEL"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Mask,
    Bias,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    DP,
    L,
    M,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_mz,
    stride_mh,
    stride_mm,
    stride_mn,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_dpz,
    stride_dph,
    stride_dpm,
    stride_dpn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvk,
    Z,
    H,
    N_CTX,
    H_DIM,
    # num_block,
    inf: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    use_mask: tl.constexpr,
    use_bias: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_b = off_hz // H
    off_h = off_hz % H

    # offset pointers for batch/head
    Q += off_b * stride_qz + off_h * stride_qh
    K += off_b * stride_kz + off_h * stride_kh
    V += off_b * stride_vz + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    DP += off_b * stride_dpz + off_h * stride_dph

    if use_bias:
        Bias += off_b * stride_bz + off_h * stride_bh
    if use_mask:
        # offs_base_mask = off_b * N_CTX
        Mask += off_b * stride_mz + off_h * stride_mh

    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    for start_n in range(0, num_block_n):
        # lo = start_n * BLOCK_M
        lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # BLOCK_M
        offs_m = tl.arange(0, BLOCK_M)  # BLOCK_N
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn)
        do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_k[None, :] * stride_dok)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)
        dp_ptrs = DP + (offs_qm[:, None] * stride_dpm + offs_n[None, :] * stride_dpn)
        if use_bias:
            b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :] * stride_bn)
        if use_mask:
            mask_ptrs = Mask + (
                offs_qm[:, None] * stride_mm + offs_n[None, :] * stride_mn
            )
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)  # BLOCK_M
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)  # BLOCK_M
        # k and v stay in SRAM throughout
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
            else:
                k = tl.load(k_ptrs, mask=offs_k[None, :] < H_DIM, other=0.0)
                v = tl.load(v_ptrs, mask=offs_k[None, :] < H_DIM, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
                v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
            else:
                k = tl.load(
                    k_ptrs,
                    mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                    other=0.0,
                )
                v = tl.load(
                    v_ptrs,
                    mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                    other=0.0,
                )
        # loop over rows
        num_block_m = tl.cdiv(N_CTX, BLOCK_M)
        for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            if EVEN_M & EVEN_HEADDIM:
                q = tl.load(q_ptrs)
            else:
                if EVEN_HEADDIM:
                    q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0)
                else:
                    q = tl.load(
                        q_ptrs,
                        mask=(offs_m_curr[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                        other=0.0,
                    )
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))

            if use_bias:
                tl.debug_barrier()  # Race condition otherwise
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
                        other=0.0,
                    ).to(tl.float32)
                qk = qk * sm_scale + bias

            if use_mask:
                # tl.debug_barrier()  # Race condition otherwise
                # qk = tl.where(offs_m_curr[:, None] >= N_CTX, float("-1e20"), qk)
                # qk = tl.where(offs_n[None, :] >= N_CTX, float("-1e20"), qk)
                # mask_data = tl.load(Mask + offs_base_mask + offs_n)
                # qk = tl.where(mask_data[None, :] == 0., float("-1e20"), qk)
                if EVEN_M & EVEN_N:
                    mask_data = tl.load(mask_ptrs).to(tl.float32)
                else:
                    mask_data = tl.load(
                        mask_ptrs,
                        mask=(offs_m_curr[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
                        other=0.0,
                    ).to(tl.float32)

                qk += tl.where(mask_data == 0.0, -inf, 0.0)
                # qk = tl.where(mask_data == 0., -inf, qk)

            m = tl.load(m_ptrs + offs_m_curr)
            if use_bias:
                p = tl.exp(qk - m[:, None])
            else:
                p = tl.exp(qk * sm_scale - m[:, None])
            # compute dv
            if EVEN_M & EVEN_HEADDIM:
                do = tl.load(do_ptrs)  # .to(tl.float32)
            else:
                do = tl.load(
                    do_ptrs,
                    mask=(offs_m_curr[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                    other=0.0,
                )

            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)

            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))

            # compute ds = p * (dp - delta[:, None])
            ds = p * dp
            if use_bias:
                tl.store(dp_ptrs, ds)
            ds = ds * sm_scale

            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)

            # compute dq
            # can we remove .to(tl.float32)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs).to(tl.float32)
                dq += tl.dot(ds.to(Q.dtype.element_ty), k)
                tl.store(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0
                    ).to(tl.float32)
                    dq += tl.dot(ds.to(Q.dtype.element_ty), k)
                    tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < N_CTX)
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                        other=0.0,
                    ).to(tl.float32)
                    dq += tl.dot(ds.to(Q.dtype.element_ty), k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                    )
            # increment pointers
            dq_ptrs += BLOCK_M * stride_dqm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_dom

            dp_ptrs += BLOCK_M * stride_dpm
            if use_bias:
                b_ptrs += BLOCK_M * stride_bm
            if use_mask:
                mask_ptrs += BLOCK_M * stride_mm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv)
                tl.store(dk_ptrs, dk)
            else:
                tl.store(dv_ptrs, dv, mask=offs_k[None, :] < H_DIM)
                tl.store(dk_ptrs, dk, mask=offs_k[None, :] < H_DIM)
        else:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)
                tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)
            else:
                tl.store(
                    dv_ptrs,
                    dv,
                    mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                )
                tl.store(
                    dk_ptrs,
                    dk,
                    mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < H_DIM),
                )
