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
import os
from torch.utils.cpp_extension import ROCM_HOME
import warnings

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

def check_if_rocm_pytorch():
    is_rocm_pytorch = False
    if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    return is_rocm_pytorch

IS_ROCM_PYTORCH = check_if_rocm_pytorch()

# an envrionment variable to explicitly switch on/off aiter backend
# by default it is 1, which means aiter backend is enabled
USE_ROCM_AITER_ROPE_BACKEND = int(os.environ.get("USE_ROCM_AITER_ROPE_BACKEND", 1)) == 1

# a flag to switch between the native apex kernel and native aiter kernel
# by default it is False
AITER_ROPE_BACKEND = False
'''
False - native kernel in apex repo
True - aiter native kernel
'''

# switch on aiter backend if it is rocm and aiter is enabled from the user
if IS_ROCM_PYTORCH and USE_ROCM_AITER_ROPE_BACKEND:
    try:
        import aiter
        AITER_ROPE_BACKEND = True
        warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
    except ImportError:
        AITER_ROPE_BACKEND = False
if not AITER_ROPE_BACKEND:
    import fused_rotary_positional_embedding
    warnings.warn("Using the native apex kernel for RoPE.", UserWarning)


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
        raise ValueError("Invalid forward implementation.")

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        raise ValueError("Invalid backward implementation.")

class FusedRoPEFuncApex(FusedRoPEFunc):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
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
        (freqs,) = ctx.saved_tensors
        grad_input = fused_rotary_positional_embedding.backward(
            grad_output, freqs, ctx.transpose_output_memory
        )
        return grad_input, None, None

class FusedRoPEFuncAiter(FusedRoPEFunc):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
        s = t.shape[0]
        b = t.shape[1]
        h = t.shape[2]
        d = t.shape[3]
        # t is of shape [s, b, h, d]
        # freqs is of shape [s, 1, 1, d]    

        act_options = {'dtype': t.dtype, 'device': t.device, 'requires_grad': False}
        if transpose_output_memory:
            output = torch.empty((b, s, h, d), **act_options).transpose(0, 1)
        else:
            output = torch.empty((s, b, h, d), **act_options)
        aiter.rope_fwd_impl(output, t, freqs, 0, False, False)

        ctx.save_for_backward(freqs)
        ctx.transpose_output_memory = transpose_output_memory

        return output

    
    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (freqs,) = ctx.saved_tensors   

        s = grad_output.shape[0]
        b = grad_output.shape[1]
        h = grad_output.shape[2]
        d = grad_output.shape[3]
        
        act_options = {'dtype': grad_output.dtype, 'device': grad_output.device, 'requires_grad': False}
        if ctx.transpose_output_memory:
            grad_input = torch.empty((b, s, h, d), **act_options).transpose(0, 1)
        else:
            grad_input = torch.empty((s, b, h, d), **act_options)
        aiter.rope_bwd_impl(grad_input, grad_output, freqs, 0, False, False)

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
    FusedRoPEFunc = FusedRoPEFuncAiter if AITER_ROPE_BACKEND else FusedRoPEFuncApex
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
        raise ValueError("Invalid forward implementation.")
            
    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        raise ValueError("Invalid backward implementation.")

class FusedRoPECachedFuncApex(FusedRoPECachedFunc):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cos_: torch.Tensor,
        sin_: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
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
        cos_, sin_ = ctx.saved_tensors
        grad_input = fused_rotary_positional_embedding.backward_cached(
            grad_output, cos_, sin_, ctx.transpose_output_memory
        )
        return grad_input, None, None, None
    
class FusedRoPECachedFuncAiter(FusedRoPECachedFunc):    
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cos_: torch.Tensor,
        sin_: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
        s = t.shape[0]
        b = t.shape[1]
        h = t.shape[2]
        d = t.shape[3]
        # t is of shape [s, b, h, d]
        # freqs is of shape [s, 1, 1, d]    

        act_options = {'dtype': t.dtype, 'device': t.device, 'requires_grad': False}
        if transpose_output_memory:
            output = torch.empty((b, s, h, d), **act_options).transpose(0, 1)
        else:
            output = torch.empty((s, b, h, d), **act_options)
        aiter.rope_cached_fwd_impl(output, t, cos_, sin_, 0, False, False)

        ctx.save_for_backward(cos_, sin_)
        ctx.transpose_output_memory = transpose_output_memory

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        cos_, sin_ = ctx.saved_tensors

        s = grad_output.shape[0]
        b = grad_output.shape[1]
        h = grad_output.shape[2]
        d = grad_output.shape[3]
        
        act_options = {'dtype': grad_output.dtype, 'device': grad_output.device, 'requires_grad': False}
        if ctx.transpose_output_memory:
            grad_input = torch.empty((b, s, h, d), **act_options).transpose(0, 1)
        else:
            grad_input = torch.empty((s, b, h, d), **act_options)
        aiter.rope_cached_bwd_impl(grad_input, grad_output, cos_, sin_, 0, False, False)
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
    FusedRoPEFunc = FusedRoPECachedFuncAiter if AITER_ROPE_BACKEND else FusedRoPECachedFuncApex
    return FusedRoPEFunc.apply(t, cos_, sin_, transpose_output_memory)

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
        raise ValueError("Invalid forward implementation.")
        
    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        raise ValueError("Invalid backward implementation.")

class FusedRoPETHDFuncApex(FusedRoPETHDFunc):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        output = fused_rotary_positional_embedding.forward_thd(
            t, cu_seqlens, freqs
        )
        ctx.save_for_backward(cu_seqlens, freqs)
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        cu_seqlens, freqs = ctx.saved_tensors
        grad_input = fused_rotary_positional_embedding.backward_thd(
            grad_output, cu_seqlens, freqs
        )
        return grad_input, None, None

class FusedRoPETHDFuncAiter(FusedRoPETHDFunc):

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        t1 = t.shape[0]
        h = t.shape[1]
        d = t.shape[2]
        # t is of shape [t, h, d]

        act_options = {'dtype': t.dtype, 'device': t.device, 'requires_grad': False}
        output = torch.empty((t1, h, d), **act_options)
        aiter.rope_thd_fwd_impl(output, t, cu_seqlens, freqs, 0, False, False)

        ctx.save_for_backward(cu_seqlens, freqs)

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        cu_seqlens, freqs = ctx.saved_tensors

        t = grad_output.shape[0]
        h = grad_output.shape[1]
        d = grad_output.shape[2]
        # t is of shape [t, h, d]
        
        act_options = {'dtype': grad_output.dtype, 'device': grad_output.device, 'requires_grad': False}
        grad_input = torch.empty((t, h, d), **act_options)
        aiter.rope_thd_bwd_impl(grad_input, grad_output, cu_seqlens, freqs, 0, False, False)

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
    FusedRoPEFunc = FusedRoPETHDFuncAiter if AITER_ROPE_BACKEND else FusedRoPETHDFuncApex
    return FusedRoPEFunc.apply(t, cu_seqlens, freqs)

class FusedRoPE2DFunc(torch.autograd.Function):
    """
    Fused 2D RoPE function
    """
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        img_h: int,
        img_w: int,
        cos_h: torch.Tensor,
        sin_h: torch.Tensor,
        cos_w: torch.Tensor,
        sin_w: torch.Tensor,
    ) -> torch.Tensor:
        raise ValueError("Invalid forward implementation.")

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        raise ValueError("Invalid backward implementation.")

class FusedRoPE2DFuncApex(FusedRoPE2DFunc):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        img_h: int,
        img_w: int,
        cos_h: torch.Tensor,
        sin_h: torch.Tensor,
        cos_w: torch.Tensor,
        sin_w: torch.Tensor,
    ) -> torch.Tensor:
        t = t.view(t.shape[0], img_h, img_w, t.shape[2], t.shape[3])
        output = fused_rotary_positional_embedding.forward_2d(
            t, cos_h, sin_h, cos_w, sin_w
        )
        ctx.save_for_backward(cos_h, sin_h, cos_w, sin_w)
        ctx.img_h = img_h
        ctx.img_w = img_w
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        cos_h, sin_h, cos_w, sin_w = ctx.saved_tensors

        grad_output = grad_output.view(
            grad_output.shape[0],
            ctx.img_h,
            ctx.img_w,
            grad_output.shape[2],
            grad_output.shape[3],
        )
        grad_input = fused_rotary_positional_embedding.backward_2d(
            grad_output, cos_h, sin_h, cos_w, sin_w
        )
        return grad_input, None, None, None, None, None, None

class FusedRoPE2DFuncAiter(FusedRoPE2DFunc):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        img_h: int,
        img_w: int,
        cos_h: torch.Tensor,
        sin_h: torch.Tensor,
        cos_w: torch.Tensor,
        sin_w: torch.Tensor,
    ) -> torch.Tensor:
        
        s = t.shape[0]
        h = t.shape[2]
        d = t.shape[3]
        # t is of shape [s, ih*iw, h, d]

        act_options = {'dtype': t.dtype, 'device': t.device, 'requires_grad': False}
        output = torch.empty((s, img_h * img_w, h, d), **act_options)
        aiter.rope_2d_fwd_impl(output, t, cos_h, sin_h, cos_w, sin_w, img_h, img_w, 0, False, False)
        ctx.save_for_backward(cos_h, sin_h, cos_w, sin_w)
        ctx.img_h = img_h
        ctx.img_w = img_w

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        cos_h, sin_h, cos_w, sin_w = ctx.saved_tensors
       
        s = grad_output.shape[0]
        h = grad_output.shape[2]
        d = grad_output.shape[3]
        # t is of shape [s, ih* iw, h, d]
        
        act_options = {'dtype': grad_output.dtype, 'device': grad_output.device, 'requires_grad': False}
        grad_input = torch.empty((s, ctx.img_h * ctx.img_w, h, d), **act_options)
        aiter.rope_2d_bwd_impl(grad_input, grad_output, cos_h, sin_h, cos_w, sin_w, ctx.img_h, ctx.img_w, 0, False, False)

        return grad_input, None, None, None, None, None, None

def fused_apply_rotary_pos_emb_2d(
    t: torch.Tensor,
    img_h: int,
    img_w: int,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T in `bshd` format, where
    b: batch size
    s: sequence length
    h: head num
    d: dim of each head

    Args:
        t (Tensor): Input tensor T is of shape [b, s, h, d]
        img_h (int): s == img_h * img_w
        img_w (int): s == img_h * img_w
        cos_h (Tensor): shape [1, H, 1, d // 2] and dtype either `float` or
            the same as `t`. H >= img_h.
        sin_h (Tensor): shape [1, H, 1, d // 2] and dtype either `float` or
            the same as `t`. H >= img_h.
        cos_w (Tensor): shape [1, W, 1, d // 2] and dtype either `float` or
            the same as `t`. W >= img_w.
        sin_w (Tensor): shape [1, W, 1, d // 2] and dtype either `float` or
            the same as `t`. W >= img_w.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    assert (
        t.size(1) == img_h * img_w
    ), "The sequence length should be equal to img_h * img_w"
    assert (
        cos_h.size() == sin_h.size()
    ), "The shape of cos_h and sin_h should be the same"
    assert (
        cos_w.size() == sin_w.size()
    ), "The shape of cos_w and sin_w should be the same"
    FusedRoPEFunc = FusedRoPE2DFuncAiter if AITER_ROPE_BACKEND else FusedRoPE2DFuncApex
    return FusedRoPEFunc.apply(t, img_h, img_w, cos_h, sin_h, cos_w, sin_w)