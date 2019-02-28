import torch

MODULE = torch

def _whitelist_only(config):
    return config == 'TENSOR_CORES_ONLY'

def fp16_funcs(config):
    return _FP16_FUNCS

def fp32_funcs(config):
    if _whitelist_only(config):
        return []
    else:
        return _FP32_FUNCS

def casts(config):
    if _whitelist_only(config):
        return []
    else:
        return _CASTS

def sequence_casts(config):
    if _whitelist_only(config):
        return []
    else:
        return _SEQUENCE_CASTS

_FP16_FUNCS = [
    # Math
    # TODO: why are these in top-level torch namespace?
    'conv1d',
    'conv2d',
    'conv3d',
    'conv_transpose1d',
    'conv_transpose2d',
    'conv_transpose3d',
    'conv_tbc',

    # BLAS
    'addmm',
    'addmv',
    'addr',
    'matmul',
    'mm',
    'mv',

]

# TODO: ban in-place versions of these in fp16
_FP32_FUNCS = [
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

    # Special reduction-like BLAS
    'addbmm',
    'baddbmm',
    'bmm',

    # Misc
    'renorm'
]

# Multi-tensor fns that may need type promotion
_CASTS = [
    # Multi-tensor math
    'addcdiv',
    'addcmul',
    'atan2',
    'cross',
    'prelu',

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

# Will possibly need to promote *all* elements of `seq`
_SEQUENCE_CASTS = [
    'cat', # torch.cat(seq, dim=0, out=None)
    'stack' # torch.stack(seq, dim=0, out=None)
]
