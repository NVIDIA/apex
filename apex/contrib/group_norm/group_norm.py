#!/usr/bin/env python
# coding: utf-8

#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

import functools
import os
import torch
import torch.nn.init as init
import group_norm_cuda
import group_norm_v2_cuda

from torch import Tensor
from torch.nn.parameter import Parameter

__all__ = ["GroupNorm"]


def one_time_warning(msg: str):
    if not hasattr(one_time_warning, "has_been_called"):
        one_time_warning.has_been_called = True
        print(f"\033[93m{msg}\033[0m")  # hightlight with yellow color


@functools.cache
def get_cc_and_sm_count(device_index: int):
    props = torch.cuda.get_device_properties(device_index)
    CC = (props.major, props.minor)
    SM_COUNT = props.multi_processor_count
    return CC, SM_COUNT


# pytorch group norm requires same input type
def torch_group_norm(x, g, w, b, eps, act=""):
    xdtype, wdtype = x.dtype, w.dtype
    if xdtype != wdtype:
        x = x.to(dtype=wdtype)
    y = torch.nn.functional.group_norm(x, g, w, b, eps)
    if act in ["silu", "swish"]:
        y = torch.nn.functional.silu(y)
    if xdtype != wdtype and y.dtype != xdtype:
        y = y.to(dtype=xdtype)
    return y


@torch.library.custom_op("apex::group_norm_nhwc_fprop", mutates_args=())
def group_norm_nhwc_fprop(
    x: torch.Tensor,
    G: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    act: str | None = None,
    passes: int = 1,
    use_group_norm_v2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # sanity check
    act = act.lower() if act else act
    assert x.is_contiguous(memory_format=torch.channels_last), "Only support NHWC layout."
    assert weight.numel() == x.shape[1], "Unexpected parameter count."
    assert bias.numel() == x.shape[1], "Unexpected parameter count."
    assert x.shape[1] % G == 0, "C % G != 0."
    assert act in [None, "", "silu", "swish"], "Unsupported activation."
    assert passes in [1, 2], "Invalid number of passes for algorithm."

    with_swish = act in ("silu", "swish")
    sm_margin = int(os.environ.get("APEX_GROUP_NORM_FPROP_SM_MARGIN", "0"))

    # enqueue fprop kernel
    if use_group_norm_v2:
        sums = torch.empty(x.shape[0] * G * 2, device=x.device)
        y = group_norm_v2_cuda.gn(
            x, weight, bias, eps, with_swish, G, mean_var_out=sums, sm_margin=sm_margin
        )
    else:
        if sm_margin:
            raise NotImplementedError("sm_margin is not supported for GroupNorm v1")
        y, sums = group_norm_cuda.forward(x, G, weight, bias, eps, passes, with_swish)
    return y, sums


@group_norm_nhwc_fprop.register_fake
def fake_group_norm_nhwc_fprop(
    x, G, weight, bias, eps, act=None, passes=1, use_group_norm_v2=False
):
    # sanity check
    act = act.lower() if act else act
    assert x.is_contiguous(memory_format=torch.channels_last), "Only support NHWC layout."
    assert weight.numel() == x.shape[1], "Unexpected parameter count."
    assert bias.numel() == x.shape[1], "Unexpected parameter count."
    assert x.shape[1] % G == 0, "C % G != 0."
    assert act in [None, "", "silu", "swish"], "Unsupported activation."
    assert passes in [1, 2], "Invalid number of passes for algorithm."

    y = torch.empty_like(x)
    sums = torch.empty(2 * x.shape[0] * G, device="cuda", dtype=torch.float32)
    return y, sums


@torch.library.custom_op("apex::group_norm_nhwc_bprop", mutates_args=())
def group_norm_nhwc_bprop(
    grad_output: torch.Tensor,
    sums: torch.Tensor,
    x: torch.Tensor,
    G: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    act: str | None = None,
    passes: int = 1,
    use_group_norm_v2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # sanity check
    if not grad_output.is_contiguous(memory_format=torch.channels_last):
        one_time_warning(
            "Warning: GroupNorm NHWC expects NHWC grad_output but it's not, "
            "thus a memory format change is introduced. "
            "This may come from the TorchInductor rule that tangents must be "
            "contiguous. Try to avoid graph break around NHWC tensors "
            "can fix this issue. (Future warning will be suppressed.)"
        )
        grad_output = grad_output.contiguous(memory_format=torch.channels_last)

    act = act.lower() if act else act
    with_swish = act in ["silu", "swish"]
    sm_margin = int(os.environ.get("APEX_GROUP_NORM_BPROP_SM_MARGIN", "0"))

    if use_group_norm_v2:
        dx, dw, db = group_norm_v2_cuda.gn_bwd(
            grad_output, x, weight, bias, sums, eps, with_swish, G, sm_margin=sm_margin
        )
    else:
        if sm_margin:
            raise NotImplementedError("sm_margin is not supported for GroupNorm v1")
        dx, dw, db = group_norm_cuda.backward(
            grad_output, sums, x, G, weight, bias, eps, passes, with_swish
        )
    return dx, dw, db


@group_norm_nhwc_bprop.register_fake
def fake_group_norm_nhwc_bprop(
    grad_output,
    sums,
    x,
    G,
    weight,
    bias,
    eps,
    act=None,
    passes=1,
    use_group_norm_v2=False,
):
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias)
    return dx, dw, db


def backward(ctx, grad_output, grad_sums):
    # retrive saved info
    x, w, b, sums = ctx.saved_tensors
    G = ctx.G
    eps = ctx.eps
    passes = ctx.passes
    act = ctx.act
    use_group_norm_v2 = ctx.use_group_norm_v2

    dx, dw, db = group_norm_nhwc_bprop(
        grad_output, sums, x, G, w, b, eps, act, passes, use_group_norm_v2
    )
    return dx, None, dw, db, None, None, None, None


def setup_context(ctx, inputs, output):
    x, G, weight, bias, eps, act, passes, use_group_norm_v2 = inputs
    y, sums = output
    # save for backward
    ctx.save_for_backward(x, weight, bias, sums)
    ctx.G = G
    ctx.eps = eps
    ctx.passes = passes
    ctx.act = act
    ctx.use_group_norm_v2 = use_group_norm_v2


group_norm_nhwc_fprop.register_autograd(backward, setup_context=setup_context)


def cuda_group_norm_nhwc_one_pass(x, G, weight, bias, eps, act=None):
    y, _ = group_norm_nhwc_fprop(x, G, weight, bias, eps, act, passes=1)
    return y


def cuda_group_norm_nhwc_two_pass(x, G, weight, bias, eps, act=None):
    y, _ = group_norm_nhwc_fprop(x, G, weight, bias, eps, act, passes=2)
    return y


def cuda_group_norm_v2_nhwc(x, G, weight, bias, eps, act=None):
    y, _ = group_norm_nhwc_fprop(x, G, weight, bias, eps, act, use_group_norm_v2=True)
    return y


# We do not direct inherit from torch.nn.GroupNorm since several fusers don't
# support inheritance. Extends:
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py
class GroupNorm(torch.nn.Module):
    """Optimized GroupNorm for NHWC layout with optional Swish/SiLU fusion.

    There are two version of CUDA kernels under the hood: one pass and two
    passes. This operator contains a simple heuristic to choose algorithm.

    Limitations:

    * Designed for 32 groups, also tested with 16 groups, some other number
      of groups can also work but not guaranteed;
    * Supported number of channels C are:

        128, 256, 320, 384, 448, 512, 640, 768, 896, 960, 1024, 1280, 1344,
        1536, 1792, 1920, 2048, 2240, 2560, 2688, 3072, 3136, 3584, 4096.

      One pass algorithm supports only channels mentioned above. Two pass
      algorithm might automatically support some other channels as well.
    * N/H/W do not have lower (except >0) and upper bound limitations;

    All the unsupported cases will be forwarded to PyTorch implementation.
    """

    __constants__ = [
        "num_groups",
        "num_channels",
        "eps",
        "affine",
        "act",
        "SUPPORTED_CHANNELS",
        "SUPPORTED_GROUPS",
    ]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    act: str | None
    SUPPORTED_CHANNELS = frozenset(
        [
            128,
            256,
            320,
            384,
            448,
            512,
            640,
            768,
            896,
            960,
            1024,
            1280,
            1344,
            1536,
            1792,
            1920,
            2048,
            2240,
            2560,
            2688,
            3072,
            3136,
            3584,
            4096,
        ]
    )
    SUPPORTED_GROUPS = frozenset([16, 32])
    SUPPORTED_DTYPES = frozenset(
        [
            # (input dtype, parameter dtype)
            (torch.float32, torch.float32),
            (torch.float32, torch.float16),
            (torch.float32, torch.bfloat16),
            (torch.float16, torch.float16),
            (torch.float16, torch.bfloat16),
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float16),
            (torch.bfloat16, torch.float32),
        ]
    )
    GN_V2_SUPPORTED_CHANNELS = frozenset(
        [
            # (HW, C)
            (8 * 8, 1280),
            (8 * 8, 2560),
            (16 * 16, 640),
            (16 * 16, 1280),
            (16 * 16, 1920),
            (16 * 16, 2560),
            (32 * 32, 320),
            (32 * 32, 640),
            (32 * 32, 960),
            (32 * 32, 1280),
            (32 * 32, 1920),
            (64 * 64, 320),
            (64 * 64, 640),
            (64 * 64, 960),
        ]
    )
    GN_V2_SUPPORTED_DTYPES = frozenset(
        [
            # (input dtype, parameter dtype)
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
        ]
    )
    GN_V2_SUPPORTED_GROUPS_SWISH = frozenset(
        [
            # (num_groups, with_swish)
            (16, True),
            (32, False),
        ]
    )
    GN_V2_SUPPORTED_LOWER_BOUND_SM_COUNT = {
        (10, 0): 148,
    }

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
        act=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.act = act.lower() if act else act
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
        sm = torch.cuda.get_device_capability(device)
        self.sm = sm[0] * 10 + sm[1]

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_legality(self, input: Tensor) -> bool:
        is_nhwc = input.is_contiguous(memory_format=torch.channels_last)
        is_legal_groups = self.num_groups in self.SUPPORTED_GROUPS
        is_legal_channels = self.num_channels in self.SUPPORTED_CHANNELS
        is_input_half_or_float_or_bf16 = input.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ]
        is_supported_dtype_combination = (
            not self.affine or (input.dtype, self.weight.dtype) in self.SUPPORTED_DTYPES
        )
        is_legal_act = self.act in [None, "", "silu", "swish"]

        if (
            is_nhwc
            and is_input_half_or_float_or_bf16
            and is_supported_dtype_combination
            and is_legal_act
            and self.affine
            and is_legal_groups
            and is_legal_channels
        ):
            return True
        else:
            return False

    def _check_v2_legality(self, input: Tensor) -> bool:
        is_legal_channels = (
            input.shape[2] * input.shape[3],
            self.num_channels,
        ) in self.GN_V2_SUPPORTED_CHANNELS
        is_supported_groups_swish_combination = (
            self.num_groups,
            self.act in ["silu", "swish"],
        ) in self.GN_V2_SUPPORTED_GROUPS_SWISH
        is_supported_dtype_combination = (
            self.affine and (input.dtype, self.weight.dtype) in self.GN_V2_SUPPORTED_DTYPES
        )
        cc, sm_count = get_cc_and_sm_count(input.device.index)
        is_supported_sm_count = (
            cc in self.GN_V2_SUPPORTED_LOWER_BOUND_SM_COUNT
            and sm_count >= self.GN_V2_SUPPORTED_LOWER_BOUND_SM_COUNT[cc]
        )

        if (
            is_legal_channels
            and is_supported_groups_swish_combination
            and is_supported_dtype_combination
            and is_supported_sm_count
        ):
            return True
        else:
            return False

    def forward(self, input: Tensor) -> Tensor:
        can_use_nhwc_group_norm = self._check_legality(input)

        if can_use_nhwc_group_norm:
            channels = input.shape[1]
            hw = 1
            for i in range(2, len(input.shape)):
                hw *= input.shape[i]
            max_hw_one_pass = 1024 if self.sm >= 80 else 256
            if (hw >= 512 and channels in (3136, 3584, 4096)) or hw > max_hw_one_pass:
                passes = 2
            else:
                passes = 1
            use_group_norm_v2 = self._check_v2_legality(input)
            y, _ = group_norm_nhwc_fprop(
                input,
                self.num_groups,
                self.weight,
                self.bias,
                self.eps,
                self.act,
                passes,
                use_group_norm_v2,
            )
            return y
        else:
            return torch_group_norm(
                input, self.num_groups, self.weight, self.bias, self.eps, self.act
            )

    def extra_repr(self) -> str:
        if self.act:
            return "{num_groups}, {num_channels}, eps={eps}, affine={affine}, act={act}".format(
                **self.__dict__
            )
        else:
            return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(
                **self.__dict__
            )
