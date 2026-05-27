import torch

from apex._custom_ops import load_custom_op_library


load_custom_op_library("_syncbn", __file__)

welford_mean_var = torch.ops.apex.syncbn_welford_mean_var
welford_parallel = torch.ops.apex.syncbn_welford_parallel
batchnorm_forward = torch.ops.apex.syncbn_batchnorm_forward
reduce_bn = torch.ops.apex.syncbn_reduce_bn
batchnorm_backward = torch.ops.apex.syncbn_batchnorm_backward
welford_mean_var_c_last = torch.ops.apex.syncbn_welford_mean_var_c_last
batchnorm_forward_c_last = torch.ops.apex.syncbn_batchnorm_forward_c_last
reduce_bn_c_last = torch.ops.apex.syncbn_reduce_bn_c_last
batchnorm_backward_c_last = torch.ops.apex.syncbn_batchnorm_backward_c_last
relu_bw_c_last = torch.ops.apex.syncbn_relu_bw_c_last
