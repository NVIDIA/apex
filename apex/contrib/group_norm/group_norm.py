#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are not permit-
# ted.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F
import torch.nn.init as init
import group_norm_cuda

from torch import Tensor
from torch.nn.parameter import Parameter
from functools import partial

__all__ = ['GroupNorm']

def one_time_warning(msg: str):
    if not hasattr(one_time_warning, 'has_been_called'):
        one_time_warning.has_been_called = True
        print(f'\033[93m{msg}\033[0m')  # hightlight with yellow color

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
    act: str | None=None,
    passes: int=1,
) -> tuple[torch.Tensor, torch.Tensor]:
    # sanity check
    act = act.lower() if act else act
    assert x.is_contiguous(memory_format=torch.channels_last), \
        "Only support NHWC layout."
    assert weight.numel() == x.shape[1], "Unexpected parameter count."
    assert bias.numel() == x.shape[1], "Unexpected parameter count."
    assert x.shape[1] % G == 0, "C % G != 0."
    assert act in [None, "", "silu", "swish"], "Unsupported activation."
    assert passes in [1, 2], "Invalid number of passes for algorithm."

    with_swish = act in ("silu", "swish")

    # enqueue fprop kernel
    y, sums = group_norm_cuda.forward(x, G, weight, bias, eps, passes,
                                          with_swish)
    return y, sums

@group_norm_nhwc_fprop.register_fake
def fake_group_norm_nhwc_fprop(x, G, weight, bias, eps, act=None, passes=1):
    # sanity check
    act = act.lower() if act else act
    assert x.is_contiguous(memory_format=torch.channels_last), \
        "Only support NHWC layout."
    assert weight.numel() == x.shape[1], "Unexpected parameter count."
    assert bias.numel() == x.shape[1], "Unexpected parameter count."
    assert x.shape[1] % G == 0, "C % G != 0."
    assert act in [None, "", "silu", "swish"], "Unsupported activation."
    assert passes in [1, 2], "Invalid number of passes for algorithm."

    y = torch.empty_like(x)
    sums = torch.empty(2 * x.shape[0] * G, device="cuda", dtype=torch.float32);
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
    act: str | None=None,
    passes: int=1,
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
    with_swish = (act in ["silu", "swish"])

    dx, dw, db = group_norm_cuda.backward(grad_output, sums, x, G,
                                          weight, bias, eps,
                                          passes, with_swish)
    return dx, dw, db

@group_norm_nhwc_bprop.register_fake
def fake_group_norm_nhwc_bprop(grad_output, sums, x, G, weight, bias, eps, act=None, passes=1):
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

    dx, dw, db = group_norm_nhwc_bprop(grad_output, sums, x, G, w, b, eps,
                                       act, passes)
    return dx, None, dw, db, None, None, None

def setup_context(ctx, inputs, output):
    x, G, weight, bias, eps, act, passes = inputs
    y, sums = output
    # save for backward
    ctx.save_for_backward(x, weight, bias, sums)
    ctx.G = G
    ctx.eps = eps
    ctx.passes = passes
    ctx.act = act

group_norm_nhwc_fprop.register_autograd(backward, setup_context=setup_context)

def cuda_group_norm_nhwc_one_pass(x, G, weight, bias, eps, act=None):
    y, _ = group_norm_nhwc_fprop(x, G, weight, bias, eps, act, passes=1)
    return y

def cuda_group_norm_nhwc_two_pass(x, G, weight, bias, eps, act=None):
    y, _ = group_norm_nhwc_fprop(x, G, weight, bias, eps, act, passes=2)
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
        'num_groups', 'num_channels', 'eps', 'affine', 'act',
        'SUPPORTED_CHANNELS', 'SUPPORTED_GROUPS'
    ]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    act: str | None
    SUPPORTED_CHANNELS = frozenset([
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
    ])
    SUPPORTED_GROUPS = frozenset([16, 32])
    SUPPORTED_DTYPES = frozenset([
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
        ])

    def __init__(self,
                 num_groups: int,
                 num_channels: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 device=None,
                 dtype=None,
                 act=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.act = act.lower() if act else act
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels,
                                                **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

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
            torch.float16, torch.bfloat16, torch.float32
        ]
        is_supported_dtype_combination = not self.affine or \
            (input.dtype, self.weight.dtype) in self.SUPPORTED_DTYPES
        is_legal_act = self.act in [None, '', 'silu', 'swish']

        if is_nhwc and is_input_half_or_float_or_bf16 and \
                is_supported_dtype_combination and is_legal_act and \
                self.affine and is_legal_groups and is_legal_channels:
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
            if (hw >= 512 and channels
                    in (3136, 3584, 4096)) or hw > max_hw_one_pass:
                passes = 2
            else:
                passes = 1
            y, _ = group_norm_nhwc_fprop(input, self.num_groups, self.weight,
                                         self.bias, self.eps, self.act, passes)
            return y
        else:
            return torch_group_norm(input, self.num_groups, self.weight,
                                    self.bias, self.eps, self.act)

    def extra_repr(self) -> str:
        if self.act:
            return '{num_groups}, {num_channels}, eps={eps}, ' \
                'affine={affine}, act={act}'.format(**self.__dict__)
        else:
            return '{num_groups}, {num_channels}, eps={eps}, ' \
                'affine={affine}'.format(**self.__dict__)
