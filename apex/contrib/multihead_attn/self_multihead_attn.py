import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .self_multihead_attn_func import self_attn_func
from .fast_self_multihead_attn_func import fast_self_attn_func
from .fast_self_multihead_attn_norm_add_func import fast_self_attn_norm_add_func
from apex.normalization.fused_layer_norm import FusedLayerNorm

@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        include_norm_add=False,
        impl="fast",
        separate_qkv_params=False,
        mask_additive=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = bias
        self.include_norm_add = include_norm_add
        self.impl = impl
        self.scaling = self.head_dim ** -0.5
        self.separate_qkv_params = separate_qkv_params
        self.mask_additive = mask_additive
        if mask_additive:
            assert self.include_norm_add == False, "additive mask not supported with layer norm"
            assert impl == "default" or (
                impl == "fast" and bias
            ), "additive mask not supported for fast mode without bias"
        if separate_qkv_params:
            self.q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.v_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        else:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if self.bias:
            if separate_qkv_params:
                self.q_bias = Parameter(torch.Tensor(embed_dim))
                self.k_bias = Parameter(torch.Tensor(embed_dim))
                self.v_bias = Parameter(torch.Tensor(embed_dim))
            else:
                self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
            self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        else:
            if separate_qkv_params:
                self.register_parameter("q_bias", None)
                self.register_parameter("k_bias", None)
                self.register_parameter("v_bias", None)
                self.q_bias = None
                self.k_bias = None
                self.v_bias = None
            else:
                self.register_parameter("in_proj_bias", None)
                self.in_proj_bias = None
            self.register_parameter("out_proj_bias", None)
            self.out_proj_bias = None
        if self.include_norm_add:
            if impl == "fast":
                self.lyr_nrm_gamma_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm_beta_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm = None
            else:
                self.register_parameter("lyr_norm_gamma_weights", None)
                self.register_parameter("lyr_norm_beta_weights", None)
                self.lyr_nrm_gamma_weights = None
                self.lyr_nrm_beta_weights = None
                self.lyr_nrm = FusedLayerNorm(embed_dim)
        self.reset_parameters()

        if self.include_norm_add:
            if impl == "fast":
                self.attn_func = fast_self_attn_norm_add_func
            elif impl == "default":
                self.attn_func = self_attn_func
            else:
                assert False, "Unsupported impl: {} !".format(impl)
        else:
            if impl == "fast":
                self.attn_func = fast_self_attn_func
            elif impl == "default":
                self.attn_func = self_attn_func
            else:
                assert False, "Unsupported impl: {} !".format(impl)

    def reset_parameters(self):
        if self.separate_qkv_params:
            nn.init.xavier_uniform_(self.q_weight)
            nn.init.xavier_uniform_(self.k_weight)
            nn.init.xavier_uniform_(self.v_weight)
        else:
            # in_proj_weight has shape [3 * hidden, hidden] but it should be
            # initialized like a [hidden, hidden] matrix.
            # sqrt(6 / (hidden + hidden)) / sqrt(6 / (3 * hidden + hidden)) = sqrt(2)
            # therefore xavier_uniform gain should be set to sqrt(2).
            nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.bias:
            if self.separate_qkv_params:
                nn.init.constant_(self.q_bias, 0.0)
                nn.init.constant_(self.k_bias, 0.0)
                nn.init.constant_(self.v_bias, 0.0)
            else:
                nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)
        if self.include_norm_add:
            if self.impl == "fast":
                nn.init.ones_(self.lyr_nrm_gamma_weights)
                nn.init.zeros_(self.lyr_nrm_beta_weights)
            else:
                self.lyr_nrm.reset_parameters()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        if self.separate_qkv_params:
            input_weights = (
                torch.cat(
                    [
                        self.q_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim),
                        self.k_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim),
                        self.v_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim),
                    ],
                    dim=1,
                )
                .reshape(3 * self.embed_dim, self.embed_dim)
                .contiguous()
            )
        else:
            input_weights = self.in_proj_weight
        if self.bias:
            if self.separate_qkv_params:
                input_bias = (
                    torch.cat(
                        [
                            self.q_bias.view(self.num_heads, 1, self.head_dim),
                            self.k_bias.view(self.num_heads, 1, self.head_dim),
                            self.v_bias.view(self.num_heads, 1, self.head_dim),
                        ],
                        dim=1,
                    )
                    .reshape(3 * self.embed_dim)
                    .contiguous()
                )
            else:
                input_bias = self.in_proj_bias
        else:
            input_bias = None
        if key_padding_mask is not None:
            assert attn_mask is None, "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
        elif attn_mask is not None:
            assert self.mask_additive == False, "additive mask not supported for time mask"
            mask = attn_mask
        else:
            mask = None

        if self.include_norm_add:
            if self.impl == "fast":
                outputs = self.attn_func(
                    attn_mask is not None,
                    is_training,
                    self.num_heads,
                    query,
                    self.lyr_nrm_gamma_weights,
                    self.lyr_nrm_beta_weights,
                    input_weights,
                    self.out_proj_weight,
                    mask,
                    self.dropout,
                )
            else:
                lyr_nrm_results = self.lyr_nrm(query)
                outputs = self.attn_func(
                    attn_mask is not None,
                    is_training,
                    self.num_heads,
                    self.scaling,
                    lyr_nrm_results,
                    input_weights,
                    self.out_proj_weight,
                    input_bias,
                    self.out_proj_bias,
                    mask,
                    self.mask_additive,
                    self.dropout,
                )
                if is_training:
                    outputs = jit_dropout_add(outputs, query, self.dropout, is_training)
                else:
                    outputs = outputs + query
        else:
            if self.impl == "fast":
                outputs = self.attn_func(
                    attn_mask is not None,
                    is_training,
                    self.num_heads,
                    query,
                    input_weights,
                    self.out_proj_weight,
                    input_bias,
                    self.out_proj_bias,
                    mask,
                    self.mask_additive,
                    self.dropout,
                )
            else:
                outputs = self.attn_func(
                    attn_mask is not None,
                    is_training,
                    self.num_heads,
                    self.scaling,
                    query,
                    input_weights,
                    self.out_proj_weight,
                    input_bias,
                    self.out_proj_bias,
                    mask,
                    self.mask_additive,
                    self.dropout,
                )

        return outputs, None
