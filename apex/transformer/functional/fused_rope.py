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
    """
    Fused RoPE function

    This implementation assumes the input tensor to be in `sbhd` format and the RoPE tensor to be
    of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid the expensive
    `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
        import fused_rotary_positional_embedding

        output = fused_rotary_positional_embedding.forward(
            t, freqs, transpose_output_memory
        )
        ctx.save_for_backward(freqs)
        ctx.transpose_output_memory = transpose_output_memory

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        import fused_rotary_positional_embedding

        (freqs,) = ctx.saved_tensors
        grad_input = fused_rotary_positional_embedding.backward(
            grad_output, freqs, ctx.transpose_output_memory
        )

        return grad_input, None, None


def fused_apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    transpose_output_memory: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T in `sbhd` format, where
    s: sequence length
    b: batch size
    h: head num
    d: dim of each head

    Args:
        t (Tensor): Input tensor T is of shape [s, b, h, d]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [s, 1, 1, d] and
        `float` dtype
        transpose_output_memory (bool): Default to False. Whether to transpose the 's' and 'b'
        dimension of the output's underlying memory format. This is very helpful when you want to
        get a contiguous tensor after calling `output.transpose(0, 1)`.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    return FusedRoPEFunc.apply(t, freqs, transpose_output_memory)


class FusedRoPECachedFunc(torch.autograd.Function):
    """
    Fused RoPE function

    This implementation assumes the input tensor to be in `sbhd` format and the RoPE tensor to be
    of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid the expensive
    `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cos_: torch.Tensor,
        sin_: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
        import fused_rotary_positional_embedding

        output = fused_rotary_positional_embedding.forward_cached(
            t, cos_, sin_, transpose_output_memory
        )
        ctx.save_for_backward(cos_, sin_)
        ctx.transpose_output_memory = transpose_output_memory

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        import fused_rotary_positional_embedding

        cos_, sin_ = ctx.saved_tensors
        grad_input = fused_rotary_positional_embedding.backward_cached(
            grad_output, cos_, sin_, ctx.transpose_output_memory
        )

        return grad_input, None, None, None


def fused_apply_rotary_pos_emb_cached(
    t: torch.Tensor,
    cos_: torch.Tensor,
    sin_: torch.Tensor,
    transpose_output_memory: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T in `sbhd` format, where
    s: sequence length
    b: batch size
    h: head num
    d: dim of each head

    Args:
        t (Tensor): Input tensor T is of shape [s, b, h, d]
        cos_ (Tensor): Cached cosine of the rotary positional embedding tensor is of
        shape [s, 1, 1, d] and dtype either `float` or the same as `t`.
        sin_ (Tensor): Cached sine of the rotary positional embedding tensor is of
        shape [s, 1, 1, d] and dtype either `float` or the same as `t`.
        transpose_output_memory (bool): Default to False. Whether to transpose the 's' and 'b'
        dimension of the output's underlying memory format. This is very helpful when you want to
        get a contiguous tensor after calling `output.transpose(0, 1)`.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    return FusedRoPECachedFunc.apply(t, cos_, sin_, transpose_output_memory)


class FusedRoPETHDFunc(torch.autograd.Function):
    """
    Fused RoPE function for `thd` format.

    This implementation accepts arbitrary memory layouts to avoid the expensive
    `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        import fused_rotary_positional_embedding

        output = fused_rotary_positional_embedding.forward_thd(
            t, cu_seqlens, freqs
        )
        ctx.save_for_backward(cu_seqlens, freqs)

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        import fused_rotary_positional_embedding

        cu_seqlens, freqs = ctx.saved_tensors
        grad_input = fused_rotary_positional_embedding.backward_thd(
            grad_output, cu_seqlens, freqs
        )

        return grad_input, None, None


def fused_apply_rotary_pos_emb_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T in `thd` format, where
    t: cumulative sum of sequence lengths
    h: head num
    d: dim of each head

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens (Tensor): Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d] and
        `float` dtype

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    return FusedRoPETHDFunc.apply(t, cu_seqlens, freqs)
