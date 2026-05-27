import torch

from apex._custom_ops import load_custom_op_library, scalar_float, scalar_int


load_custom_op_library("_scaled_masked_softmax_cuda", __file__)


def forward(input, mask, scale_factor):
    return torch.ops.apex.scaled_masked_softmax_forward(input, mask, scalar_float(scale_factor))


def backward(output_grads, softmax_results, scale_factor):
    return torch.ops.apex.scaled_masked_softmax_backward(output_grads, softmax_results, scalar_float(scale_factor))


def get_batch_per_block(query_seq_len, key_seq_len, batches, attn_heads):
    return torch.ops.apex.scaled_masked_softmax_get_batch_per_block(
        scalar_int(query_seq_len), scalar_int(key_seq_len), scalar_int(batches), scalar_int(attn_heads)
    )
