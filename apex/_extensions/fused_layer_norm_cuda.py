import torch

from apex._custom_ops import load_custom_op_library


load_custom_op_library("_fused_layer_norm_cuda", __file__)

forward_affine = torch.ops.apex.fused_layer_norm_forward_affine
forward = torch.ops.apex.fused_layer_norm_forward
backward_affine = torch.ops.apex.fused_layer_norm_backward_affine
backward = torch.ops.apex.fused_layer_norm_backward
forward_affine_mixed_dtypes = torch.ops.apex.fused_layer_norm_forward_affine_mixed_dtypes
rms_forward_affine = torch.ops.apex.fused_layer_norm_rms_forward_affine
rms_forward = torch.ops.apex.fused_layer_norm_rms_forward
rms_backward_affine = torch.ops.apex.fused_layer_norm_rms_backward_affine
rms_backward = torch.ops.apex.fused_layer_norm_rms_backward
rms_forward_affine_mixed_dtypes = torch.ops.apex.fused_layer_norm_rms_forward_affine_mixed_dtypes
