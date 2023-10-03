# © 2023 NVIDIA CORPORATION & AFFILIATES

import torch
import triton
import triton.language as tl
from triton import Config

# %% Constants for efficient memory access.
CACHE_SECTOR_SIZE = 32 * 8
BF16_LOAD_SIZE = int(CACHE_SECTOR_SIZE / torch.finfo(torch.bfloat16).bits)
PARTIAL_REDUCE_MIN = 32


# %% Separated backward kernels for contiguous inputs. We choose to not fusing them because dX and
# d{W, b} reduce along different directions.
@triton.autotune(
    configs=[
        Config({"M_BLOCK": 1}, num_warps=1),
        Config({"M_BLOCK": 2}, num_warps=1),
        Config({"M_BLOCK": 4}, num_warps=2),
        Config({"M_BLOCK": 8}, num_warps=4),
        Config({"M_BLOCK": 16}, num_warps=8),
        Config({"M_BLOCK": 32}, num_warps=8),
        Config({"M_BLOCK": 64}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={
        "N_BLOCK": lambda kwargs: triton.next_power_of_2(kwargs["N"]),
    },
)
@triton.jit
def _layer_norm_backward_dx(
    dy_ptr,
    x_ptr,
    w_ptr,
    x_invstd_ptr,
    x_mean_ptr,
    dx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    m_idx = (tl.program_id(0) * M_BLOCK + tl.arange(0, M_BLOCK))[:, None]
    m_mask = m_idx < M
    n_idx = tl.arange(0, N_BLOCK)[None, :]
    n_mask = n_idx < N
    mask = m_mask & n_mask
    x = tl.load(x_ptr + N * m_idx + n_idx, mask, other=0).to(tl.float32)
    x_mean = tl.load(x_mean_ptr + m_idx, m_mask, other=0).to(tl.float32)
    x_invstd = tl.load(x_invstd_ptr + m_idx, m_mask, other=0).to(tl.float32)
    x_hat = (x - x_mean) * x_invstd
    dy = tl.load(dy_ptr + N * m_idx + n_idx, mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + n_idx, n_mask, other=0).to(tl.float32)
    c1 = tl.sum(x_hat * dy * w, axis=1) / N
    c2 = tl.sum(dy * w, axis=1) / N
    dx = x_invstd * (dy * w - c1[:, None] * x_hat - c2[:, None])
    tl.store(dx_ptr + N * m_idx + n_idx, dx, mask)


@triton.autotune(
    configs=[
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN},
            num_warps=2,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 2},
            num_warps=4,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 4},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 8},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 16},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN},
            num_warps=4,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 2},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 4},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 8},
            num_warps=8,
        ),
        Config(
            {
                "N_BLOCK": BF16_LOAD_SIZE * 2,
                "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 16,
            },
            num_warps=8,
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def _layer_norm_backward_dw_db_partial(
    dy_ptr,
    x_ptr,
    x_invstd_ptr,
    x_mean_ptr,
    dw_partial_buf_ptr,
    db_partial_buf_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BUF_N_STRIDE: tl.constexpr,
    N_BLOCK: tl.constexpr,
    M_PARTIAL_REDUCE: tl.constexpr,
):
    m_idx = (tl.program_id(0) * M_PARTIAL_REDUCE + tl.arange(0, M_PARTIAL_REDUCE))[
        :, None
    ]
    m_mask = m_idx < M
    n_idx = tl.program_id(1) * N_BLOCK + tl.arange(0, N_BLOCK)
    n_mask = n_idx < N
    idx = N * m_idx + n_idx[None, :]
    mask = m_mask & n_mask[None, :]
    x = tl.load(x_ptr + idx, mask, other=0).to(tl.float32)
    x_mean = tl.load(x_mean_ptr + m_idx, m_mask, other=0).to(tl.float32)
    x_invstd = tl.load(x_invstd_ptr + m_idx, m_mask, other=0).to(tl.float32)
    x_hat = (x - x_mean) * x_invstd
    dy = tl.load(dy_ptr + idx, mask, other=0).to(tl.float32)
    dw_partial = tl.sum(dy * x_hat, axis=0)
    db_partial = tl.sum(dy, axis=0)
    tl.store(
        dw_partial_buf_ptr + BUF_N_STRIDE * n_idx + tl.program_id(0), dw_partial, n_mask
    )
    tl.store(
        db_partial_buf_ptr + BUF_N_STRIDE * n_idx + tl.program_id(0), db_partial, n_mask
    )


# %% Backward kernels for noncontiguous inputs. Using similar strided access logic as in forward.
@triton.autotune(
    configs=[
        Config({"M_BLOCK": 1}, num_warps=1),
        Config({"M_BLOCK": 2}, num_warps=1),
        Config({"M_BLOCK": 4}, num_warps=2),
        Config({"M_BLOCK": 8}, num_warps=4),
        Config({"M_BLOCK": 16}, num_warps=8),
        Config({"M_BLOCK": 32}, num_warps=8),
        Config({"M_BLOCK": 64}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={
        "N_BLOCK": lambda kwargs: triton.next_power_of_2(kwargs["N"]),
    },
)
@triton.jit
def _layer_norm_backward_dx_strided(
    dy_ptr,
    x_ptr,
    w_ptr,
    x_invstd_ptr,
    x_mean_ptr,
    dx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    m_logic_idx = tl.program_id(0) * M_BLOCK + tl.arange(0, M_BLOCK)
    m_mask = m_logic_idx < M
    m_logic_idx_0 = m_logic_idx // (D1 * D2) % D0
    m_logic_idx_1 = m_logic_idx // D2 % D1
    m_logic_idx_2 = m_logic_idx % D2
    m_idx = m_logic_idx_0 * S0 + m_logic_idx_1 * S1 + m_logic_idx_2 * S2
    n_logic_idx = tl.arange(0, N_BLOCK)
    n_mask = n_logic_idx < N
    n_idx = n_logic_idx * S3
    mask = m_mask[:, None] & n_mask[None, :]
    x_idx = m_idx[:, None] + n_idx[None, :]
    x = tl.load(x_ptr + x_idx, mask, other=0).to(tl.float32)
    x_mean = tl.load(x_mean_ptr + m_logic_idx, m_mask, other=0).to(tl.float32)[:, None]
    x_invstd = tl.load(x_invstd_ptr + m_logic_idx, m_mask, other=0).to(tl.float32)[
        :, None
    ]
    x_hat = (x - x_mean) * x_invstd
    dy_idx = N * m_logic_idx[:, None] + n_logic_idx[None, :]
    dy = tl.load(dy_ptr + dy_idx, mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + n_logic_idx, n_mask, other=0).to(tl.float32)[None, :]
    c1 = tl.sum(x_hat * dy * w, axis=1) / N
    c2 = tl.sum(dy * w, axis=1) / N
    dx = x_invstd * (dy * w - c1[:, None] * x_hat - c2[:, None])
    tl.store(dx_ptr + x_idx, dx, mask)


@triton.autotune(
    configs=[
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN},
            num_warps=2,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 2},
            num_warps=4,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 4},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 8},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 16},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN},
            num_warps=4,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 2},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 4},
            num_warps=8,
        ),
        Config(
            {"N_BLOCK": BF16_LOAD_SIZE * 2, "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 8},
            num_warps=8,
        ),
        Config(
            {
                "N_BLOCK": BF16_LOAD_SIZE * 2,
                "M_PARTIAL_REDUCE": PARTIAL_REDUCE_MIN * 16,
            },
            num_warps=8,
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def _layer_norm_backward_dw_db_partial_strided(
    dy_ptr,
    x_ptr,
    x_invstd_ptr,
    x_mean_ptr,
    dw_partial_buf_ptr,
    db_partial_buf_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BUF_N_STRIDE: tl.constexpr,
    N_BLOCK: tl.constexpr,
    M_PARTIAL_REDUCE: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    m_logic_idx = tl.program_id(0) * M_PARTIAL_REDUCE + tl.arange(0, M_PARTIAL_REDUCE)
    m_mask = m_logic_idx < M
    m_logic_idx_0 = m_logic_idx // (D1 * D2) % D0
    m_logic_idx_1 = m_logic_idx // D2 % D1
    m_logic_idx_2 = m_logic_idx % D2
    m_idx = m_logic_idx_0 * S0 + m_logic_idx_1 * S1 + m_logic_idx_2 * S2
    n_logic_idx = tl.program_id(1) * N_BLOCK + tl.arange(0, N_BLOCK)
    n_mask = n_logic_idx < N
    n_idx = n_logic_idx * S3
    mask = m_mask[:, None] & n_mask[None, :]
    x_idx = m_idx[:, None] + n_idx[None, :]
    x = tl.load(x_ptr + x_idx, mask, other=0).to(tl.float32)
    x_mean = tl.load(x_mean_ptr + m_logic_idx, m_mask, other=0).to(tl.float32)[:, None]
    x_invstd = tl.load(x_invstd_ptr + m_logic_idx, m_mask, other=0).to(tl.float32)[
        :, None
    ]
    x_hat = (x - x_mean) * x_invstd
    dy_idx = N * m_logic_idx[:, None] + n_logic_idx[None, :]
    dy = tl.load(dy_ptr + dy_idx, mask, other=0).to(tl.float32)
    dw_partial = tl.sum(dy * x_hat, axis=0)
    db_partial = tl.sum(dy, axis=0)
    tl.store(
        dw_partial_buf_ptr + BUF_N_STRIDE * n_logic_idx + tl.program_id(0),
        dw_partial,
        n_mask,
    )
    tl.store(
        db_partial_buf_ptr + BUF_N_STRIDE * n_logic_idx + tl.program_id(0),
        db_partial,
        n_mask,
    )


# %% Reduce partial accumulator buffers along the row dimension. Straightforward.
@triton.jit
def _layer_norm_backward_buf_reduce(
    partial_buf_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    N_STRIDE: tl.constexpr,
    M_STRIDE: tl.constexpr,
):
    idx = N_STRIDE * tl.program_id(0) + M_STRIDE * tl.arange(0, M)
    mask = tl.program_id(0) < N
    x = tl.sum(tl.load(partial_buf_ptr + idx, mask, other=0).to(tl.float32), axis=0)
    tl.store(output_ptr + tl.program_id(0), x, mask)
