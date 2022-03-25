###############################################################################
# Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


import torch
import torch.nn.functional as F
import fmhalib as mha

class FMHAFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cu_seqlens, p_dropout, max_s, is_training, zero_tensors):
        batch_size = cu_seqlens.numel() - 1
        if batch_size < 4:
            max_s = 512
            context, S_dmask = mha.fwd_nl(qkv, cu_seqlens, p_dropout, max_s, is_training, True, zero_tensors, None)
        else:
            context, S_dmask = mha.fwd(qkv, cu_seqlens, p_dropout, max_s, is_training, False, zero_tensors, None)
        ctx.save_for_backward(qkv, S_dmask)
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.zero_tensors = zero_tensors
        return context
    
    @staticmethod
    def backward(ctx, dout):
        qkv, S_dmask = ctx.saved_tensors
        batch_size = ctx.cu_seqlens.numel() - 1
        if batch_size < 4:
            dqkv, dp, _ = mha.bwd_nl(dout, qkv, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s, ctx.zero_tensors)
        else:
            dqkv, dp = mha.bwd(dout, qkv, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s, ctx.zero_tensors)

        return dqkv, None, None, None, None, None

class FMHA(torch.nn.Module):

    def __init__(self, config):

        super(FMHA, self).__init__()

        self.p_dropout = config.attention_probs_dropout_prob
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d = self.hidden_size // self.h
        assert self.d * self.h == self.hidden_size, "Invalid hidden size/num_heads"

    def forward(self, qkv, cu_seqlens, max_s, is_training=True, zero_tensors=False):

        ctx = FMHAFun.apply(qkv.view(-1, 3, self.h, self.d), cu_seqlens, self.p_dropout, max_s, is_training, zero_tensors)

        return ctx.view(-1, self.hidden_size)
