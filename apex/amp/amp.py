from . import compat, utils, wrap
from .handle import AmpHandle, NoOpHandle
from .lists import functional_overrides, torch_overrides, tensor_overrides

import functools
import itertools

import torch

_DECORATOR_HANDLE = None
_USER_CAST_REGISTRY = set()
_USER_PROMOTE_REGISTRY = set()

def _decorator_helper(orig_fn, cast_fn, wrap_fn):
    def wrapper(*args, **kwargs):
        handle = _DECORATOR_HANDLE
        if handle is None or not handle.is_active():
            return orig_fn(*args, **kwargs)
        inner_cast_fn = utils.verbosify(cast_fn, orig_fn.__name__,
                                  handle.verbose)
        return wrap_fn(orig_fn, inner_cast_fn, handle)(*args, **kwargs)
    return wrapper

# Decorator form
def half_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=True)
    return _decorator_helper(fn, utils.maybe_half, wrap_fn)

def float_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=False)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)

def promote_function(fn):
    wrap_fn = functools.partial(wrap.make_promote_wrapper)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)

# Registry form
def register_half_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_half))

def register_float_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_float))

def register_promote_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_PROMOTE_REGISTRY.add((mod, fn.__name__))

# Top-level function to insert _all_ the hooks.
def build(enabled=True, enable_caching=True, verbose=False):
    global _DECORATOR_HANDLE

    if not enabled:
        handle = NoOpHandle()
        _DECORATOR_HANDLE = handle
        return handle

    handle = AmpHandle(enable_caching, verbose)

    # 0) Force-{fp16, fp32} for user-annotated functions
    for mod, fn, cast_fn in _USER_CAST_REGISTRY:
        try_caching = (cast_fn == utils.maybe_half)
        wrap.cached_cast(mod, fn, cast_fn, handle,
                         try_caching, verbose)
    _USER_CAST_REGISTRY.clear()

    # 0.5) Force-promote for user-annotated functions
    for mod, fn in _USER_PROMOTE_REGISTRY:
        wrap.promote(mod, fn, verbose)
    _USER_PROMOTE_REGISTRY.clear()

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

    _DECORATOR_HANDLE = handle
    return handle
