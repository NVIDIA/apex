import torch

from apex._custom_ops import load_custom_op_library, scalar_float


load_custom_op_library("_xentropy_cuda", __file__)

__version__ = torch.ops.apex.xentropy_version()


def forward(input, labels, smoothing, half_to_float):
    return torch.ops.apex.xentropy_forward(input, labels, scalar_float(smoothing), half_to_float)


def backward(grad_loss, logits, max_log_sum_exp, labels, smoothing):
    return torch.ops.apex.xentropy_backward(
        grad_loss, logits, max_log_sum_exp, labels, scalar_float(smoothing)
    )
