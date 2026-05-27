import torch

from apex._custom_ops import load_custom_op_library


load_custom_op_library("_fused_rotary_positional_embedding", __file__)

forward = torch.ops.apex.fused_rope_forward
backward = torch.ops.apex.fused_rope_backward
forward_cached = torch.ops.apex.fused_rope_forward_cached
backward_cached = torch.ops.apex.fused_rope_backward_cached
forward_thd = torch.ops.apex.fused_rope_forward_thd
backward_thd = torch.ops.apex.fused_rope_backward_thd
forward_2d = torch.ops.apex.fused_rope_forward_2d
backward_2d = torch.ops.apex.fused_rope_backward_2d
