# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple, Union
import torch


class FusedRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, t: torch.Tensor, cos_: torch.Tensor, sin_: torch.Tensor
    ) -> torch.Tensor:
        import fused_rotary_positional_embedding

        output = fused_rotary_positional_embedding.forward(t, cos_, sin_)
        ctx.save_for_backward(cos_, sin_)

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        import fused_rotary_positional_embedding

        cos_, sin_ = ctx.saved_tensors
        grad_q = fused_rotary_positional_embedding.backward(grad_output, cos_, sin_)

        return grad_q, None, None


def fused_apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T.

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    return FusedRoPEFunc.apply(t, cos_, sin_)


def fused_apply_rotary_pos_emb_cached(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T.

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        cos (Tensor): Cached cosine of the rotary positional embedding tensor is of shape [seq_length, ..., dim]
        sin (Tensor): Cached sine of the rotary positional embedding tensor is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    cos_ = cos.to(t.dtype)
    sin_ = sin.to(t.dtype)
    return FusedRoPEFunc.apply(t, cos_, sin_)
