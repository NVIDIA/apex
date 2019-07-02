from .. import compat
from . import torch_overrides

import importlib

import torch

# if compat.variable_is_tensor() and not compat.tensor_is_variable():
MODULE = torch.Tensor
# else:
#     MODULE = torch.autograd.Variable


FP16_FUNCS = [
    '__matmul__',
]

FP32_FUNCS = [
    '__ipow__',
    '__pow__',
    '__rpow__',

    # Cast to fp32 before transfer to CPU
    'cpu',
]

CASTS = [
    '__add__',
    '__div__',
    '__eq__',
    '__ge__',
    '__gt__',
    '__iadd__',
    '__idiv__',
    '__imul__',
    '__isub__',
    '__itruediv__',
    '__le__',
    '__lt__',
    '__mul__',
    '__ne__',
    '__radd__',
    '__rdiv__',
    '__rmul__',
    '__rsub__',
    '__rtruediv__',
    '__sub__',
    '__truediv__',
]

# None of these, but here to make code cleaner.
SEQUENCE_CASTS = []

# We need to grab all the methods from torch_overrides and add them to
# the Tensor lists as well, as almost all methods are duplicated
# between `torch` and `torch.Tensor` (and check with `hasattr`,
# because a few random ones aren't defined on Tensor)
_self_mod = importlib.import_module(__name__)
for attrname in ['FP16_FUNCS', 'FP32_FUNCS', 'CASTS', 'SEQUENCE_CASTS']:
    lst = getattr(_self_mod, attrname)
    for fn in getattr(torch_overrides, attrname):
        if hasattr(MODULE, fn):
            lst.append(fn)
