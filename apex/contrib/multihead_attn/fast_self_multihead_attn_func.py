import torch

import fast_multihead_attn


class FastSelfAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        use_time_mask,
        is_training,
        heads,
        inputs,
        input_weights,
        output_weights,
        input_biases,
        output_biases,
        pad_mask,
        mask_additive,
        dropout_prob,
    ):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        mask_additive_t = torch.tensor([mask_additive])

        if use_biases_t[0]:
            if not mask_additive:
                (
                    input_lin_results,
                    softmax_results,
                    dropout_results,
                    dropout_mask,
                    matmul2_results,
                    outputs,
                ) = fast_multihead_attn.self_attn_bias_forward(
                    use_mask,
                    use_time_mask,
                    is_training,
                    heads,
                    inputs,
                    input_weights,
                    output_weights,
                    input_biases,
                    output_biases,
                    pad_mask if use_mask else null_tensor,
                    dropout_prob,
                )
                # fast_self_multihead_attn_bias.forward()                           \
                ctx.save_for_backward(
                    use_biases_t,
                    heads_t,
                    matmul2_results,
                    dropout_results,
                    softmax_results,
                    null_tensor,
                    null_tensor,
                    mask_additive_t,
                    input_lin_results,
                    inputs,
                    input_weights,
                    output_weights,
                    dropout_mask,
                    dropout_prob_t,
                )

            else:
                (
                    input_lin_results,
                    bmm1_results,
                    dropout_results,
                    dropout_mask,
                    matmul2_results,
                    outputs,
                ) = fast_multihead_attn.self_attn_bias_additive_mask_forward(
                    use_mask,
                    use_time_mask,
                    is_training,
                    heads,
                    inputs,
                    input_weights,
                    output_weights,
                    input_biases,
                    output_biases,
                    pad_mask if use_mask else null_tensor,
                    dropout_prob,
                )
                # fast_self_multihead_attn_bias_additive_mask.forward(                           \
                ctx.save_for_backward(
                    use_biases_t,
                    heads_t,
                    matmul2_results,
                    dropout_results,
                    null_tensor,
                    bmm1_results,
                    pad_mask,
                    mask_additive_t,
                    input_lin_results,
                    inputs,
                    input_weights,
                    output_weights,
                    dropout_mask,
                    dropout_prob_t,
                )

        else:
            (
                input_lin_results,
                softmax_results,
                dropout_results,
                dropout_mask,
                matmul2_results,
                outputs,
            ) = fast_multihead_attn.self_attn_forward(
                use_mask,
                use_time_mask,
                is_training,
                heads,
                inputs,
                input_weights,
                output_weights,
                pad_mask if use_mask else null_tensor,
                dropout_prob,
            )
            # fast_self_multihead_attn.forward(                           \
            ctx.save_for_backward(
                use_biases_t,
                heads_t,
                matmul2_results,
                dropout_results,
                softmax_results,
                null_tensor,
                null_tensor,
                mask_additive_t,
                input_lin_results,
                inputs,
                input_weights,
                output_weights,
                dropout_mask,
                dropout_prob_t,
            )
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (
            use_biases_t,
            heads_t,
            matmul2_results,
            dropout_results,
            softmax_results,
            bmm1_results,
            pad_mask,
            mask_additive_t,
            input_lin_results,
            inputs,
            input_weights,
            output_weights,
            dropout_mask,
            dropout_prob_t,
        ) = ctx.saved_tensors

        if use_biases_t[0]:
            if not mask_additive_t[0]:
                (
                    input_grads,
                    input_weight_grads,
                    output_weight_grads,
                    input_bias_grads,
                    output_bias_grads,
                ) = fast_multihead_attn.self_attn_bias_backward(
                    heads_t[0],
                    output_grads,
                    matmul2_results,
                    dropout_results,
                    softmax_results,
                    input_lin_results,
                    inputs,
                    input_weights,
                    output_weights,
                    dropout_mask,
                    dropout_prob_t[0],
                )
                # fast_self_multihead_attn_bias.backward(                          \

            else:
                (
                    input_grads,
                    input_weight_grads,
                    output_weight_grads,
                    input_bias_grads,
                    output_bias_grads,
                ) = fast_multihead_attn.self_attn_bias_additive_mask_backward(
                    heads_t[0],
                    output_grads,
                    matmul2_results,
                    dropout_results,
                    bmm1_results,
                    pad_mask,
                    input_lin_results,
                    inputs,
                    input_weights,
                    output_weights,
                    dropout_mask,
                    dropout_prob_t[0],
                )
                # fast_self_multihead_attn_bias_additive_mask.backward(                          \

        else:
            input_bias_grads = None
            output_bias_grads = None
            input_grads, input_weight_grads, output_weight_grads = fast_multihead_attn.self_attn_backward(
                heads_t[0],
                output_grads,
                matmul2_results,
                dropout_results,
                softmax_results,
                input_lin_results,
                inputs,
                input_weights,
                output_weights,
                dropout_mask,
                dropout_prob_t[0],
            )
            # fast_self_multihead_attn.backward(                          \
        return (
            None,
            None,
            None,
            input_grads,
            input_weight_grads,
            output_weight_grads,
            input_bias_grads,
            output_bias_grads,
            None,
            None,
            None,
        )


fast_self_attn_func = FastSelfAttnFunc.apply
