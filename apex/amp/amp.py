from . import compat, utils, wrap
from .handle import AmpHandle, NoOpHandle
from .lists import functional_overrides, torch_overrides, tensor_overrides

import inspect
import itertools

import torch

_USER_REGISTRY = set()

# Can be used as a @decorator directly on the fn
# or called w/ arg by user before `enable()`
def register_half(fn):
    mod = inspect.getmodule(fn)
    _USER_REGISTRY.add((mod, fn.__name__, utils.maybe_half))
    return fn

def register_float(fn):
    mod = inspect.getmodule(fn)
    _USER_REGISTRY.add((mod, fn.__name__, utils.maybe_float))
    return fn

# Top-level function to insert _all_ the hooks.
def build(enabled=True, enable_caching=True, verbose=False):
    if not enabled:
        return NoOpHandle()

    handle = AmpHandle(enable_caching)

    # 0) Force-{fp16, fp32} for user-annotated functions
    for mod, fn, cast_fn in _USER_REGISTRY:
        try_caching = (cast_fn == utils.maybe_half)
        wrap.cached_cast(mod, fn, cast_fn, handle,
                         try_caching, verbose)
    _USER_REGISTRY.clear()

    # 1) Force-{fp16, fp32} on white- / black-list functions
    override_modules = [functional_overrides,
                        torch_overrides,
                        tensor_overrides]
    cast_table = [('FP16_FUNCS', utils.maybe_half),
                  ('FP32_FUNCS', utils.maybe_float)]
    for module, (list_name, cast_fn) in itertools.product(override_modules,
                                                          cast_table):
        for fn in getattr(module, list_name):
            try_caching = (cast_fn == utils.maybe_half)
            wrap.cached_cast(module.MODULE, fn, cast_fn, handle,
                             try_caching, verbose)

    # 1.5) Pre-0.4, put the blacklist methods on HalfTensor and whitelist
    #      methods on FloatTensor, since they're distinct types.
    if compat.tensor_is_float_tensor():
        for fn in tensor_overrides.FP16_FUNCS:
            wrap.cached_cast(torch.cuda.FloatTensor, fn, utils.maybe_half,
                             handle, try_caching=True, verbose=verbose)
        for fn in tensor_overrides.FP32_FUNCS:
            wrap.cached_cast(torch.cuda.HalfTensor, fn, utils.maybe_float,
                             handle, try_caching=False, verbose=verbose)

    # 2) Enable type-promotion on multi-arg functions and methods.
    #    NB: special handling for sequence fns (e.g. `torch.cat`).
    promote_modules = [torch_overrides, tensor_overrides]
    promote_table = [('CASTS', wrap.promote),
                     ('SEQUENCE_CASTS', wrap.sequence_promote)]
    for promote_mod, (list_name, promote_fn) in itertools.product(promote_modules,
                                                                  promote_table):
        for fn in getattr(promote_mod, list_name):
            promote_fn(promote_mod.MODULE, fn, verbose)

    # 2.5) Pre-0.4, add blacklist methods directly to HalfTensor and FloatTensor types
    if compat.tensor_is_float_tensor():
        for cls, (list_name, promote_fn) in itertools.product([torch.cuda.FloatTensor,
                                                               torch.cuda.HalfTensor],
                                                              promote_table):
            for fn in getattr(tensor_overrides, list_name):
                promote_fn(cls, fn, verbose)

    # 3) For any in-place version of a blacklist function, error if any input is fp16.
    #    NB: this is overly conservative.
    for fn in utils.as_inplace(torch_overrides.FP32_FUNCS):
        wrap.err_if_any_half(torch_overrides.MODULE, fn)

    # 3.5) For any in-place blacklist method, error if called on fp16 tensor
    for fn in utils.as_inplace(tensor_overrides.FP32_FUNCS):
        wrap.err_if_arg0_half(tensor_overrides.MODULE, fn, verbose)
        if compat.tensor_is_float_tensor():
            wrap.err_if_arg0_half(torch.cuda.HalfTensor, fn, verbose)

    # 4) For other in-place methods, match the type of self tensor
    for fn in utils.as_inplace(itertools.chain(
            tensor_overrides.FP16_FUNCS,
            tensor_overrides.CASTS)):
        wrap.promote_match_arg0(tensor_overrides.MODULE, fn, verbose)
        if compat.tensor_is_float_tensor():
            wrap.promote_match_arg0(torch.cuda.HalfTensor, fn, verbose)
            wrap.promote_match_arg0(torch.cuda.FloatTensor, fn, verbose)

    # 5) Special handling to whitelist RNN cell backend impls.
    for fn in ['RNNReLUCell', 'RNNTanhCell', 'LSTMCell', 'GRUCell']:
        wrap.cached_cast(torch.nn.backends.thnn.backend, fn, utils.maybe_half,
                         handle, try_caching=True, verbose=verbose)

    # 5.5) Extra-special handling of RNN backend
    wrap.rnn_cast(torch.nn.backends.thnn.backend, 'RNN', verbose)

    return handle
