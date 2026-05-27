import torch

from apex._custom_ops import load_custom_op_library


load_custom_op_library("_fused_index_mul_2d", __file__)

float_forward = torch.ops.apex.index_mul_2d_float_forward
float_backward = torch.ops.apex.index_mul_2d_float_backward
float_backward_backward = torch.ops.apex.index_mul_2d_float_backward_backward
half_forward = torch.ops.apex.index_mul_2d_half_forward
half_backward = torch.ops.apex.index_mul_2d_half_backward
half_backward_backward = torch.ops.apex.index_mul_2d_half_backward_backward
