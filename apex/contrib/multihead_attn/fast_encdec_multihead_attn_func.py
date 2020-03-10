import torch
import fast_encdec_multihead_attn


class FastEncdecAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, pad_mask, dropout_prob):
        heads_t        = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor    = torch.tensor([])
        use_mask       = (pad_mask is not None)

        input_lin_q_results,                                            \
        input_lin_kv_results,                                           \
        softmax_results,                                                \
        dropout_results,                                                \
        dropout_mask,                                                   \
        matmul2_results,                                                \
        outputs =                                                       \
            fast_encdec_multihead_attn.forward(                         \
                              use_mask,                                 \
                              use_time_mask,                            \
                              is_training,                              \
                              heads,                                    \
                              inputs_q,                                 \
                              inputs_kv,                                \
                              input_weights_q,                          \
                              input_weights_kv,                         \
                              output_weights,                           \
                              pad_mask if use_mask else null_tensor,    \
                              dropout_prob)

        ctx.save_for_backward(heads_t,                                  \
                              matmul2_results,                          \
                              dropout_results,                          \
                              softmax_results,                          \
                              input_lin_q_results,                      \
                              input_lin_kv_results,                     \
                              inputs_q,                                 \
                              inputs_kv,                                \
                              input_weights_q,                          \
                              input_weights_kv,                         \
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
        input_lin_q_results,                                            \
        input_lin_kv_results,                                           \
        inputs_q,                                                       \
        inputs_kv,                                                      \
        input_weights_q,                                                \
        input_weights_kv,                                               \
        output_weights,                                                 \
        dropout_mask,                                                   \
        dropout_prob_t      = ctx.saved_tensors

        input_q_grads,                                                  \
        input_kv_grads,                                                 \
        input_weight_q_grads,                                           \
        input_weight_kv_grads,                                          \
        output_weight_grads =                                           \
            fast_encdec_multihead_attn.backward(                        \
                              heads_t[0],                               \
                              output_grads,                             \
                              matmul2_results,                          \
                              dropout_results,                          \
                              softmax_results,                          \
                              input_lin_q_results,                      \
                              input_lin_kv_results,                     \
                              inputs_q,                                 \
                              inputs_kv,                                \
                              input_weights_q,                          \
                              input_weights_kv,                         \
                              output_weights,                           \
                              dropout_mask,                             \
                              dropout_prob_t[0])

        return None, None, None, input_q_grads, input_kv_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads, None, None

fast_encdec_attn_func = FastEncdecAttnFunc.apply
