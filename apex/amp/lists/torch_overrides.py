import torch

from .. import utils

MODULE = torch

FP16_FUNCS = [
    # Low level functions wrapped by torch.nn layers.
    # The wrapper layers contain the weights which are then passed in as a parameter
    # to these functions.
    'conv1d',
    'conv2d',
    'conv3d',
    'conv_transpose1d',
    'conv_transpose2d',
    'conv_transpose3d',
    'conv_tbc',
    'prelu',

    # BLAS
    'addmm',
    'addmv',
    'addr',
    'matmul',
    'mm',
    'mv',
]

FP32_FUNCS = [
    # Pointwise
    'acos',
    'asin',
    'cosh',
    'erfinv',
    'exp',
    'expm1',
    'log',
    'log10',
    'log2',
    'reciprocal',
    'rsqrt',
    'sinh',
    'tan',

    # Other math
    'pow',

    # Reduction
    'cumprod',
    'cumsum',
    'dist',
    'mean',
    'norm',
    'prod',
    'std',
    'sum',
    'var',

    # Misc
    'renorm'
]

# Before CUDA 9.1, batched matmul was missing fast FP16 kernels. We
# check the CUDA version -- if at least 9.1, then put the bmm
# functions on the fp16 list. Otherwise, put them on the fp32 list.
_bmms = ['addbmm',
         'baddbmm',
         'bmm']
if utils.get_cuda_version() >= (9, 1, 0):
    FP16_FUNCS.extend(_bmms)
else:
    FP32_FUNCS.extend(_bmms)

# Multi-tensor fns that may need type promotion
CASTS = [
    # Multi-tensor math
    'addcdiv',
    'addcmul',
    'atan2',
    'cross',

    # Element-wise _or_ tensor-wise math
    'add',
    'div',
    'mul',

    # Comparison
    'eq',
    'equal',
    'ge',
    'gt',
    'le',
    'lt',
    'ne'
]

# Functions that take sequence arguments. We need to inspect the whole
# sequence and cast to the widest type.
SEQUENCE_CASTS = [
    'cat',
    'stack'
]
