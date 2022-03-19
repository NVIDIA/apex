# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

import fast_multihead_attn


class FastEncdecAttnNormAddFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        use_time_mask,
        is_training,
        heads,
        inputs_q,
        inputs_kv,
        lyr_nrm_gamma_weights,
        lyr_nrm_beta_weights,
        input_weights_q,
        input_weights_kv,
        output_weights,
        pad_mask,
        dropout_prob,
    ):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None

        (
            lyr_nrm_results,
            lyr_nrm_mean,
            lyr_nrm_invvar,
            input_lin_q_results,
            input_lin_kv_results,
            softmax_results,
            dropout_results,
            dropout_mask,
            matmul2_results,
            dropout_add_mask,
            outputs,
        ) = fast_multihead_attn.encdec_multihead_attn_norm_add_forward(
            use_mask,
            use_time_mask,
            is_training,
            heads,
            inputs_q,
            inputs_kv,
            lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights,
            input_weights_q,
            input_weights_kv,
            output_weights,
            pad_mask if use_mask else null_tensor,
            dropout_prob,
        )

        ctx.save_for_backward(
            heads_t,
            matmul2_results,
            dropout_results,
            softmax_results,
            input_lin_q_results,
            input_lin_kv_results,
            lyr_nrm_results,
            lyr_nrm_mean,
            lyr_nrm_invvar,
            inputs_q,
            inputs_kv,
            lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights,
            input_weights_q,
            input_weights_kv,
            output_weights,
            dropout_mask,
            dropout_add_mask,
            dropout_prob_t,
        )

        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (
            heads_t,
            matmul2_results,
            dropout_results,
            softmax_results,
            input_lin_q_results,
            input_lin_kv_results,
            lyr_nrm_results,
            lyr_nrm_mean,
            lyr_nrm_invvar,
            inputs_q,
            inputs_kv,
            lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights,
            input_weights_q,
            input_weights_kv,
            output_weights,
            dropout_mask,
            dropout_add_mask,
            dropout_prob_t,
        ) = ctx.saved_tensors

        (
            input_q_grads,
            input_kv_grads,
            lyr_nrm_gamma_grads,
            lyr_nrm_beta_grads,
            input_weight_q_grads,
            input_weight_kv_grads,
            output_weight_grads,
        ) = fast_multihead_attn.encdec_multihead_attn_norm_add_backward(
            heads_t[0],
            output_grads,
            matmul2_results,
            dropout_results,
            softmax_results,
            input_lin_q_results,
            input_lin_kv_results,
            lyr_nrm_results,
            lyr_nrm_mean,
            lyr_nrm_invvar,
            inputs_q,
            inputs_kv,
            lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights,
            input_weights_q,
            input_weights_kv,
            output_weights,
            dropout_mask,
            dropout_add_mask,
            dropout_prob_t[0],
        )

        # import pdb; pdb.set_trace()
        return (
            None,
            None,
            None,
            input_q_grads,
            input_kv_grads,
            lyr_nrm_gamma_grads,
            lyr_nrm_beta_grads,
            input_weight_q_grads,
            input_weight_kv_grads,
            output_weight_grads,
            None,
            None,
        )


fast_encdec_attn_norm_add_func = FastEncdecAttnNormAddFunc.apply
