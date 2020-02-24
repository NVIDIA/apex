import torch
import fast_self_multihead_attn


class FastSelfAttnFunc(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs, input_weights, output_weights, pad_mask, dropout_prob):
        heads_t        = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor    = torch.tensor([])
        use_mask       = (pad_mask is not None)

        input_lin_results,                                              \
        softmax_results,                                                \
        dropout_results,                                                \
        dropout_mask,                                                   \
        matmul2_results,                                                \
        outputs =                                                       \
            fast_self_multihead_attn.forward(                           \
                              use_mask,                                 \
                              use_time_mask,                            \
                              is_training,                              \
                              heads,                                    \
                              inputs,                                   \
                              input_weights,                            \
                              output_weights,                           \
                              pad_mask if use_mask else null_tensor,    \
                              dropout_prob)

        ctx.save_for_backward(heads_t,                                  \
                              matmul2_results,                          \
                              dropout_results,                          \
                              softmax_results,                          \
                              input_lin_results,                        \
                              inputs,                                   \
                              input_weights,                            \
                              output_weights,                           \
                              dropout_mask,                             \
                              dropout_prob_t)

        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        heads_t,                                                        \
        matmul2_results,                                                \
        dropout_results,                                                \
        softmax_results,                                                \
        input_lin_results,                                              \
        inputs,                                                         \
        input_weights,                                                  \
        output_weights,                                                 \
        dropout_mask,                                                   \
        dropout_prob_t      = ctx.saved_tensors

        input_grads,                                                    \
        input_weight_grads,                                             \
        output_weight_grads =                                           \
            fast_self_multihead_attn.backward(                          \
                              heads_t[0],                               \
                              output_grads,                             \
                              matmul2_results,                          \
                              dropout_results,                          \
                              softmax_results,                          \
                              input_lin_results,                        \
                              inputs,                                   \
                              input_weights,                            \
                              output_weights,                           \
                              dropout_mask,                             \
                              dropout_prob_t[0])

        return None, None, None, input_grads, input_weight_grads, output_weight_grads, None, None

fast_self_attn_func = FastSelfAttnFunc.apply
