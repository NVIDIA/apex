# © 2023 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

from collections import defaultdict
from enum import Enum, unique
from itertools import chain
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.optim import Adam, Optimizer

# The most common parameter size in open-fold.
CHUNK_SIZE = torch.tensor(128, dtype=torch.int64)


# Data type enumerates. tl.constexpr arg doesn't accept Triton data types.
@unique
class _DTypeEnum(Enum):
    FP16 = 0
    BF16 = 1
    FP32 = 2
    FP64 = 3


_TORCH2DTYPE = {
    torch.float16: _DTypeEnum.FP16,
    torch.bfloat16: _DTypeEnum.BF16,
    torch.float32: _DTypeEnum.FP32,
    torch.float64: _DTypeEnum.FP64,
}


_DTYPE2TRITON = {
    _DTypeEnum.FP16: tl.float16,
    _DTypeEnum.BF16: tl.bfloat16,
    _DTypeEnum.FP32: tl.float32,
    _DTypeEnum.FP64: tl.float64,
}


# Adam math impl enumerates. There're minor impl differences between Apex and official PyTorch.
@unique
class AdamMathType(Enum):
    ApexAdam = 0
    ApexAdamW = 1
    PyTorchAdam = 2


@triton.jit
def _adam_math(
    param,
    grad,
    moment,
    velocity,
    beta1,
    beta2,
    beta1_correction,
    beta2_correction,
    eps,
    lr,
    weight_decay,
    adam_math_mode: tl.constexpr,
):
    if adam_math_mode == tl.constexpr(AdamMathType.ApexAdam.value):
        grad += weight_decay * param
        moment *= beta1
        moment += (1.0 - beta1) * grad
        velocity *= beta2
        velocity += (1.0 - beta2) * grad * grad
        update = (moment / beta1_correction) / (tl.math.sqrt(velocity / beta2_correction) + eps)
        param -= lr * update
    elif adam_math_mode == tl.constexpr(AdamMathType.ApexAdamW.value):
        moment *= beta1
        moment += (1.0 - beta1) * grad
        velocity *= beta2
        velocity += (1.0 - beta2) * grad * grad
        update = (moment / beta1_correction) / (tl.math.sqrt(velocity / beta2_correction) + eps)
        update += weight_decay * param
        param -= lr * update
    elif adam_math_mode == tl.constexpr(AdamMathType.PyTorchAdam.value):
        grad += weight_decay * param
        moment *= beta1
        moment += (1.0 - beta1) * grad
        velocity *= beta2
        velocity += (1.0 - beta2) * grad * grad
        # PyTorch computes step_size and denominator separately so it can use addcdiv later.
        step_size = -lr / beta1_correction
        beta2_correction_sqrt = tl.math.sqrt(beta2_correction)
        denom = tl.math.sqrt(velocity) / beta2_correction_sqrt + eps
        param += step_size * (moment / denom)
    else:
        raise ValueError(f"Unknown Adam math mode: {adam_math_mode}")
    return param, moment, velocity


# OpenFold model doesn't use buffers, so only update parameters.
@triton.jit
def _swa_math(
    param,
    swa_param,
    decay_rate,
    n_averaged,
):
    if n_averaged == 0:
        swa_param = param
    else:
        swa_param += (1.0 - decay_rate) * (param - swa_param)
    return swa_param


@triton.jit
def _multi_tensor_adam_swa(
    state_param_ptr_per_chunk,
    compute_param_ptr_per_chunk,
    swa_param_ptr_per_chunk,
    grad_ptr_per_chunk,
    moment_ptr_per_chunk,
    velocity_ptr_per_chunk,
    chunk_local_idx_ptr,
    chunk_numel_ptr,
    grad_clip_scale_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    beta1_correction,
    beta2_correction,
    swa_decay_rate,
    swa_n_averaged,
    adam_math_mode: tl.constexpr,
    MODEL_COMPUTE_DTYPE: tl.constexpr,
    MODEL_STATE_DTYPE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    chunk_local_idx = tl.load(chunk_local_idx_ptr + chunk_idx)
    chunk_numel = tl.load(chunk_numel_ptr + chunk_idx)

    compute_dtype = _DTYPE2TRITON[MODEL_COMPUTE_DTYPE.value]
    compute_pointer_type = tl.pointer_type(compute_dtype)
    state_dtype = _DTYPE2TRITON[MODEL_STATE_DTYPE.value]
    state_pointer_type = tl.pointer_type(state_dtype)

    state_param_ptr = tl.load(state_param_ptr_per_chunk + chunk_idx).to(state_pointer_type)
    swa_param_ptr = tl.load(swa_param_ptr_per_chunk + chunk_idx).to(state_pointer_type)
    moment_ptr = tl.load(moment_ptr_per_chunk + chunk_idx).to(state_pointer_type)
    velocity_ptr = tl.load(velocity_ptr_per_chunk + chunk_idx).to(state_pointer_type)
    compute_param_ptr = tl.load(compute_param_ptr_per_chunk + chunk_idx).to(compute_pointer_type)
    grad_ptr = tl.load(grad_ptr_per_chunk + chunk_idx).to(compute_pointer_type)
    grad_clip_scale = tl.load(grad_clip_scale_ptr)

    ptr_base_offset = chunk_local_idx * CHUNK_SIZE
    state_param_ptr += ptr_base_offset
    compute_param_ptr += ptr_base_offset
    swa_param_ptr += ptr_base_offset
    grad_ptr += ptr_base_offset
    moment_ptr += ptr_base_offset
    velocity_ptr += ptr_base_offset

    for i in range(0, CHUNK_SIZE, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < chunk_numel
        # Gradient clip step.
        grad = tl.load(grad_ptr + idx, mask).to(state_dtype)
        grad *= grad_clip_scale
        # Adam step.
        param = tl.load(state_param_ptr + idx, mask)
        moment = tl.load(moment_ptr + idx, mask)
        velocity = tl.load(velocity_ptr + idx, mask)
        param, moment, velocity = _adam_math(
            param=param,
            grad=grad,
            moment=moment,
            velocity=velocity,
            beta1=beta1,
            beta2=beta2,
            beta1_correction=beta1_correction,
            beta2_correction=beta2_correction,
            eps=eps,
            lr=lr,
            weight_decay=weight_decay,
            adam_math_mode=adam_math_mode,
        )
        # SWA step.
        swa_param = tl.load(swa_param_ptr + idx, mask)
        swa_param = _swa_math(
            param=param,
            swa_param=swa_param,
            decay_rate=swa_decay_rate,
            n_averaged=swa_n_averaged,
        )
        # Write results. BF16 and SWA parameters are updated as well.
        tl.store(state_param_ptr + idx, param, mask)
        tl.store(moment_ptr + idx, moment, mask)
        tl.store(velocity_ptr + idx, velocity, mask)
        tl.store(compute_param_ptr + idx, param, mask)
        tl.store(swa_param_ptr + idx, swa_param, mask)


# Note:
# - Gradients are attached to BF16 tensors
# - Assume all parameters are all updated at each step, i.e., they share the same step number
class FusedAdamSWA(Optimizer):
    def __init__(
        self,
        params: List[nn.Parameter],
        compute_params: List[nn.Parameter],
        swa_params: List[nn.Parameter],
        swa_decay_rate: float,
        lr: float = 1e-3,
        bias_correction: bool = True,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        adam_math_mode: AdamMathType = AdamMathType.PyTorchAdam,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        set_grad_none: bool = True,
        capturable: bool = False,
        master_weights: bool = False,
    ):
        if not isinstance(params, list):
            params = list(params)
        if not isinstance(compute_params, list):
            compute_params = list(compute_params)
        if not isinstance(swa_params, list):
            swa_params = list(swa_params)
        if not compute_params or not swa_params:
            raise ValueError("FusedAdamSWA requires both BF16 and SWA parameters.")
        if not len(params) == len(compute_params) == len(swa_params):
            raise ValueError(
                "FusedAdamSWA expects params, bf16_params, and swa_params to have same length"
            )
        if not all(
            p.shape == b.shape == s.shape for p, b, s in zip(params, compute_params, swa_params)
        ):
            raise ValueError(
                "FusedAdamSWA expects each state in params, bf16_params, abd swa_params to have same shape"
            )
        if not all(p.dtype == s.dtype for p, s in zip(params, swa_params)):
            raise ValueError("FusedAdamSWA expects all params and swa_params to have same dtype")
        if not all(p.is_contiguous() for p in chain(params, compute_params, swa_params)):
            raise ValueError("FusedAdamSWA expects all input params to be contiguous")
        if amsgrad:
            raise NotImplementedError("amsgrad is not supported by FusedAdamSWA")
        if capturable:
            raise NotImplementedError("capturable is not supported by FusedAdamSWA")
        if master_weights:
            raise NotImplementedError("master_weights is not supported by FusedAdamSWA")
        if not isinstance(adam_math_mode, AdamMathType):
            raise ValueError(
                f"Unknown Adam math mode {adam_math_mode}, expect to be any of:\n"
                f"\t- {AdamMathType.ApexAdam}: NVIDIA Apex Adam math;\n"
                f"\t- {AdamMathType.ApexAdamW}: NVIDIA Apex Adam math with adam_w set to True;\n"
                f"\t- {AdamMathType.PyTorchAdam}: The official PyTorch Adam math.\n"
            )

        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.adam_math_mode = adam_math_mode
        self.set_grad_none = set_grad_none
        self.compute_param_groups = [{"params": compute_params}]
        self.swa_param_groups = [{"params": swa_params, "n_averaged": 0}]
        self.swa_decay_rate = swa_decay_rate

        # We assume that parameter and buffer pointers won't change throughout the training, only
        # gradients could be re-allocated due to set_grad_none.
        self._pointer_buffers_initialized = False

    def _build_pointer_buffers(self):
        # Loading checkpoint to optimizer re-allocates param and states, so pointer logic should be
        # at the first step of training, where we assume all states are ready.
        if not all(
            len(pg) == 1
            for pg in (
                self.param_groups,
                self.compute_param_groups,
                self.swa_param_groups,
            )
        ):
            raise RuntimeError("FusedAdamSWA does not support multiple param groups")

        # `bf16_params` contains both BF16 and FP32 data types, thus we have to group parameters
        # and other states into different buffers and launch respective kernels.
        params, compute_params, swa_params = (
            self.param_groups[0]["params"],
            self.compute_param_groups[0]["params"],
            self.swa_param_groups[0]["params"],
        )
        self.pointer_buffer_groups = defaultdict(dict)
        for i, p in enumerate(compute_params):
            compute_dtype = p.dtype
            state_dtype = params[i].dtype
            self.pointer_buffer_groups[(compute_dtype, state_dtype)].setdefault("tensor_idx", [])
            self.pointer_buffer_groups[(compute_dtype, state_dtype)]["tensor_idx"].append(i)

        for (_, state_dtype), buffer_group in self.pointer_buffer_groups.items():
            # Select tensors by dtype.
            t_idx = buffer_group["tensor_idx"]
            params_this_group = [params[i] for i in t_idx]
            compute_params_this_group = [compute_params[i] for i in t_idx]
            swa_params_this_group = [swa_params[i] for i in t_idx]

            # Build parameter pointer buffers.
            param_ptrs = torch.tensor([p.data_ptr() for p in params_this_group], dtype=torch.int64)
            compute_param_ptrs = torch.tensor(
                [b.data_ptr() for b in compute_params_this_group], dtype=torch.int64
            )
            swa_param_ptrs = torch.tensor(
                [s.data_ptr() for s in swa_params_this_group], dtype=torch.int64
            )

            param_numels = torch.tensor([p.numel() for p in params_this_group], dtype=torch.int64)
            chunks_per_param = param_numels.float().div_(CHUNK_SIZE).ceil_().long()
            chunk_local_idx = torch.cat(
                [torch.arange(chunks, dtype=torch.int64) for chunks in chunks_per_param]
            )
            chunk_numel = torch.minimum(
                param_numels.repeat_interleave(chunks_per_param) - chunk_local_idx * CHUNK_SIZE,
                CHUNK_SIZE,
            )
            param_ptr_per_chunk = torch.repeat_interleave(param_ptrs, chunks_per_param)
            compute_param_ptr_per_chunk = torch.repeat_interleave(
                compute_param_ptrs, chunks_per_param
            )
            swa_param_ptr_per_chunk = torch.repeat_interleave(swa_param_ptrs, chunks_per_param)

            device = params_this_group[0].device
            buffer_group["device"] = device
            buffer_group["chunks_per_param"] = chunks_per_param
            buffer_group["chunk_local_idx"] = chunk_local_idx.to(device)
            buffer_group["chunk_numel"] = chunk_numel.to(device)
            buffer_group["param_ptr_per_chunk"] = param_ptr_per_chunk.to(device)
            buffer_group["compute_param_ptr_per_chunk"] = compute_param_ptr_per_chunk.to(device)
            buffer_group["swa_param_ptr_per_chunk"] = swa_param_ptr_per_chunk.to(device)
            buffer_group["total_chunks"] = chunks_per_param.sum().item()
            buffer_group["default_grad_clip_scale"] = torch.tensor(1.0, dtype=state_dtype).to(
                device
            )

            # Build moment pointer buffers.
            moment, velocity = [], []
            for p in params_this_group:
                state = self.state[p]
                if "exp_avg" not in state or "exp_avg_sq" not in state:
                    state["exp_avg"] = torch.zeros_like(p.detach(), dtype=state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p.detach(), dtype=state_dtype)
                moment.append(state["exp_avg"].data_ptr())
                velocity.append(state["exp_avg_sq"].data_ptr())
            moment = torch.tensor(moment, dtype=torch.int64)
            velocity = torch.tensor(velocity, dtype=torch.int64)
            buffer_group["exp_avg_ptr_per_chunk"] = torch.repeat_interleave(
                moment, chunks_per_param
            ).to(device)
            buffer_group["exp_avg_sq_ptr_per_chunk"] = torch.repeat_interleave(
                velocity, chunks_per_param
            ).to(device)

        self._pointer_buffers_initialized = True

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        grad_clip_scale: Optional[Union[torch.Tensor, float]] = None,
    ):
        if not self._pointer_buffers_initialized:
            self._build_pointer_buffers()

        loss = closure() if closure is not None else None

        group = self.param_groups[0]
        compute_group = self.compute_param_groups[0]
        swa_group = self.swa_param_groups[0]
        if "step" in group:
            group["step"] += 1
        else:
            group["step"] = 1
        (beta1, beta2), step = group["betas"], group["step"]
        if group["bias_correction"]:
            beta1_correction = 1.0 - beta1**step
            beta2_correction = 1.0 - beta2**step
        else:
            beta1_correction = beta2_correction = 1.0

        grad_ptr = []
        for p in compute_group["params"]:
            if p.grad is None:
                continue
            if p.grad.detach().is_sparse:
                raise RuntimeError(
                    "FusedAdamSWA does not support sparse gradients, please consider SparseAdam instead"
                )
            grad_ptr.append(p.grad.data_ptr())

        for (compute_dtype, state_dtype), buffer_group in self.pointer_buffer_groups.items():
            device = buffer_group["device"]
            t_idx = buffer_group["tensor_idx"]
            grad_ptr_this_group = [grad_ptr[i] for i in t_idx]
            grad_ptr_this_group = torch.tensor(grad_ptr_this_group, dtype=torch.int64)
            grad_ptr_per_chunk = torch.repeat_interleave(
                grad_ptr_this_group, buffer_group["chunks_per_param"]
            ).to(device, non_blocking=True)
            if grad_clip_scale is None:
                grad_clip_scale_this_group = buffer_group["default_grad_clip_scale"]
            elif not torch.is_tensor(grad_clip_scale):
                grad_clip_scale_this_group = torch.tensor(grad_clip_scale).to(
                    device, non_blocking=True
                )
            else:
                grad_clip_scale_this_group = grad_clip_scale

            grid = (buffer_group["total_chunks"],)
            _multi_tensor_adam_swa[grid](
                state_param_ptr_per_chunk=buffer_group["param_ptr_per_chunk"],
                compute_param_ptr_per_chunk=buffer_group["compute_param_ptr_per_chunk"],
                swa_param_ptr_per_chunk=buffer_group["swa_param_ptr_per_chunk"],
                grad_ptr_per_chunk=grad_ptr_per_chunk,
                moment_ptr_per_chunk=buffer_group["exp_avg_ptr_per_chunk"],
                velocity_ptr_per_chunk=buffer_group["exp_avg_sq_ptr_per_chunk"],
                chunk_local_idx_ptr=buffer_group["chunk_local_idx"],
                chunk_numel_ptr=buffer_group["chunk_numel"],
                grad_clip_scale_ptr=grad_clip_scale_this_group,
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                beta1_correction=beta1_correction,
                beta2_correction=beta2_correction,
                swa_decay_rate=self.swa_decay_rate,
                swa_n_averaged=swa_group["n_averaged"],
                adam_math_mode=self.adam_math_mode.value,
                MODEL_COMPUTE_DTYPE=_TORCH2DTYPE[compute_dtype],
                MODEL_STATE_DTYPE=_TORCH2DTYPE[state_dtype],
                # TODO: Find optimal hyper-parameters.
                CHUNK_SIZE=CHUNK_SIZE.item(),
                BLOCK_SIZE=128,
                num_warps=1,
            )

        swa_group["n_averaged"] += 1

        return loss

    @classmethod
    def from_optim(
        cls,
        adam_optimizer: Adam,
        fp32_params: List[nn.Parameter],
        bf16_params: List[nn.Parameter],
        swa_params: List[nn.Parameter],
        swa_decay_rate: float,
    ) -> FusedAdamSWA:
        assert len(adam_optimizer.param_groups) == 1
        param_group = adam_optimizer.param_groups[0]
        lr = param_group["lr"]
        betas = param_group["betas"]
        eps = param_group["eps"]
        weight_decay = param_group["weight_decay"]
        amsgrad = param_group["amsgrad"]
        fused_adam_swa_optimizer = cls(
            params=fp32_params,
            compute_params=bf16_params,
            swa_params=swa_params,
            swa_decay_rate=swa_decay_rate,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            adam_math_mode=AdamMathType.PyTorchAdam,
        )
        adam_state_dict = adam_optimizer.state_dict()
        adam_state_dict["param_groups"][0].setdefault("bias_correction", True)
        steps = [v["step"] for v in adam_state_dict["state"].values()]
        if len(steps) == 0:  # Did not load optimizer checkpoint.
            steps = [torch.tensor(1)]
        elif not all(s == steps[0] for s in steps):
            raise ValueError("FusedAdamSWA requires all parameters were updated by same steps!")
        step = int(steps[0].item())
        adam_state_dict["param_groups"][0].setdefault("step", step)
        fused_adam_swa_optimizer.load_state_dict(adam_state_dict)
        return fused_adam_swa_optimizer
