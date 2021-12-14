import torch

import fast_multihead_attn


class MaskSoftmaxDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, is_training, heads, inputs, pad_mask, mask_additive, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        use_mask_t = torch.tensor([use_mask])
        mask_additive_t = torch.tensor([mask_additive])

        if mask_additive:
            dropout_results, dropout_mask, softmax_results = fast_multihead_attn.additive_mask_softmax_dropout_forward(
                use_mask, is_training, heads, inputs, pad_mask if use_mask else null_tensor, dropout_prob
            )
            # fast_additive_mask_softmax_dropout.forward(                           \
        else:
            dropout_results, dropout_mask, softmax_results = fast_multihead_attn.mask_softmax_dropout_forward(
                use_mask, is_training, heads, inputs, pad_mask if use_mask else null_tensor, dropout_prob
            )
            # fast_mask_softmax_dropout.forward(                           \

        ctx.save_for_backward(
            use_mask_t,
            heads_t,
            softmax_results,
            dropout_mask,
            pad_mask if use_mask else null_tensor,
            mask_additive_t,
            dropout_prob_t,
        )

        return dropout_results.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (
            use_mask_t,
            heads_t,
            softmax_results,
            dropout_mask,
            pad_mask,
            mask_additive_t,
            dropout_prob_t,
        ) = ctx.saved_tensors

        if mask_additive_t[0]:
            input_grads = fast_multihead_attn.additive_mask_softmax_dropout_backward(
                use_mask_t[0], heads_t[0], output_grads, softmax_results, dropout_mask, dropout_prob_t[0]
            )
            # fast_additive_mask_softmax_dropout.backward(                          \
        else:
            input_grads = fast_multihead_attn.mask_softmax_dropout_backward(
                use_mask_t[0], heads_t[0], output_grads, softmax_results, dropout_mask, pad_mask, dropout_prob_t[0]
            )
            # fast_mask_softmax_dropout.backward(                          \
        return None, None, input_grads, None, None, None


fast_mask_softmax_dropout_func = MaskSoftmaxDropout.apply
