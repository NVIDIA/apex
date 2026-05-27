import torch

from apex._custom_ops import load_custom_op_library


load_custom_op_library("_fused_dense_cuda", __file__)

linear_bias_forward = torch.ops.apex.fused_dense_linear_bias_forward
linear_bias_backward = torch.ops.apex.fused_dense_linear_bias_backward
linear_gelu_linear_forward = torch.ops.apex.fused_dense_linear_gelu_linear_forward
linear_gelu_linear_backward = torch.ops.apex.fused_dense_linear_gelu_linear_backward
