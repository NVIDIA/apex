import torch

# True for post-0.4, when Variables/Tensors merged.
def variable_is_tensor():
    v = torch.autograd.Variable()
    return isinstance(v, torch.Tensor)

def tensor_is_variable():
    x = torch.Tensor()
    return type(x) == torch.autograd.Variable

# False for post-0.4
def tensor_is_float_tensor():
    x = torch.Tensor()
    return type(x) == torch.FloatTensor

# Akin to `torch.is_tensor`, but returns True for Variable
# objects in pre-0.4.
def is_tensor_like(x):
    return torch.is_tensor(x) or isinstance(x, torch.autograd.Variable)

# Wraps `torch.is_floating_point` if present, otherwise checks
# the suffix of `x.type()`.
def is_floating_point(x):
    if hasattr(torch, 'is_floating_point'):
        return torch.is_floating_point(x)
    try:
        torch_type = x.type()
        return torch_type.endswith('FloatTensor') or \
            torch_type.endswith('HalfTensor') or \
            torch_type.endswith('DoubleTensor')
    except AttributeError:
        return False

def scalar_python_val(x):
    if hasattr(x, 'item'):
        return x.item()
    else:
        if isinstance(x, torch.autograd.Variable):
            return x.data[0]
        else:
            return x[0]

# Accounts for the possibility that some ops may be removed from a namespace.
def filter_attrs(module, attrs):
    return list(attrname for attrname in attrs if hasattr(module, attrname))
