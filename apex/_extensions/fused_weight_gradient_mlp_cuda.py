import torch

from apex._custom_ops import load_custom_op_library


load_custom_op_library("_fused_weight_gradient_mlp_cuda", __file__)

wgrad_gemm_accum_fp32 = torch.ops.apex.fused_weight_gradient_mlp_wgrad_gemm_accum_fp32
wgrad_gemm_accum_fp16 = torch.ops.apex.fused_weight_gradient_mlp_wgrad_gemm_accum_fp16
