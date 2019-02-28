from . import compat, rnn_compat, utils, wrap
from .handle import AmpHandle, NoOpHandle
from .lists import functional_overrides, torch_overrides, tensor_overrides
from ._amp_state import _amp_state
from .frontend import *

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
    _USER_PROMOTE_REGISTRY.add((module, name))


# Top-level function to insert _all_ the hooks.

def init(enabled=True, loss_scale='dynamic', enable_caching=True,
         verbose=False, allow_banned=False, config='FULL'):
    global _DECORATOR_HANDLE

    if not enabled:
        handle = NoOpHandle()
        _DECORATOR_HANDLE = handle
        return handle

    handle = AmpHandle(loss_scale, enable_caching, verbose)

    # 0) Force-{fp16, fp32} for user-annotated functions
    for mod, fn, cast_fn in _USER_CAST_REGISTRY:
        try_caching = (cast_fn == utils.maybe_half)
        wrap.cached_cast(mod, fn, cast_fn, handle,
                         try_caching, verbose)
    _USER_CAST_REGISTRY.clear()

    # 0.5) Force-promote for user-annotated functions
    for mod, fn in _USER_PROMOTE_REGISTRY:
        wrap.promote(mod, fn, handle, verbose)
    _USER_PROMOTE_REGISTRY.clear()

    # TC-only mode casts outputs from whitelist back to fp32
    whitelist_post_fn = None
    if config == 'TENSOR_CORES_ONLY':
        whitelist_post_fn = utils.maybe_float

    # 1) Force-{fp16, fp32} on white- / black-list functions
    override_modules = [functional_overrides,
                        torch_overrides,
                        tensor_overrides]
    for module in override_modules:
        for fn in module.fp16_funcs(config):
            wrap.cached_cast(module.MODULE, fn, utils.maybe_half, handle,
                             try_caching=True, verbose=verbose,
                             post_fn=whitelist_post_fn)
        for fn in module.fp32_funcs(config):
            wrap.cached_cast(module.MODULE, fn, utils.maybe_float, handle,
                             try_caching=False, verbose=verbose,
                             post_fn=None)

    # 1.5) Pre-0.4, put the blacklist methods on HalfTensor and whitelist
    #      methods on FloatTensor, since they're distinct types.
    if compat.tensor_is_float_tensor():
        for fn in tensor_overrides.fp16_funcs(config):
            wrap.cached_cast(torch.cuda.FloatTensor, fn, utils.maybe_half,
                             handle, try_caching=True, verbose=verbose)
        for fn in tensor_overrides.fp32_funcs(config):
            wrap.cached_cast(torch.cuda.HalfTensor, fn, utils.maybe_float,
                             handle, try_caching=False, verbose=verbose)

    # 2) Enable type-promotion on multi-arg functions and methods.
    #    NB: special handling for sequence fns (e.g. `torch.cat`).
    promote_modules = [torch_overrides, tensor_overrides]
    promote_table = [('casts', wrap.promote),
                     ('sequence_casts', wrap.sequence_promote)]
    for promote_mod, (list_name, promote_fn) in itertools.product(promote_modules,
                                                                  promote_table):
        for fn in getattr(promote_mod, list_name)(config):
            promote_fn(promote_mod.MODULE, fn, handle, verbose)

    # 2.5) Pre-0.4, add blacklist methods directly to HalfTensor and FloatTensor types
    if compat.tensor_is_float_tensor():
        for cls, (list_name, promote_fn) in itertools.product([torch.cuda.FloatTensor,
                                                               torch.cuda.HalfTensor],
                                                              promote_table):
            for fn in getattr(tensor_overrides, list_name)(config):
                promote_fn(cls, fn, handle, verbose)

    # 3) For any in-place version of a blacklist function, error if any input is fp16.
    #    NB: this is overly conservative.
    for fn in utils.as_inplace(torch_overrides.fp32_funcs(config)):
        wrap.err_if_any_half(torch_overrides.MODULE, fn, handle)

    # 3.5) For any in-place blacklist method, error if called on fp16 tensor
    for fn in utils.as_inplace(tensor_overrides.fp32_funcs(config)):
        wrap.err_if_arg0_half(tensor_overrides.MODULE, fn, handle, verbose)
        if compat.tensor_is_float_tensor():
            wrap.err_if_arg0_half(torch.cuda.HalfTensor, fn, handle, verbose)

    # 4) For other in-place methods, match the type of self tensor
    for fn in utils.as_inplace(itertools.chain(
            tensor_overrides.fp16_funcs(config),
            tensor_overrides.casts(config))):
        wrap.promote_match_arg0(tensor_overrides.MODULE, fn, handle, verbose)
        if compat.tensor_is_float_tensor():
            wrap.promote_match_arg0(torch.cuda.HalfTensor, fn, handle, verbose)
            wrap.promote_match_arg0(torch.cuda.FloatTensor, fn, handle, verbose)

    # 5) RNNs + RNN cells are whitelisted specially
    if rnn_compat.has_old_rnns():
        wrap.rnn_cast(torch.nn.backends.thnn.backend, 'RNN', handle, verbose,
                      post_fn=whitelist_post_fn)
    if not rnn_compat.has_old_rnns():
        # Patch in our own indirection of `_VF` in modules/rnn s.t. it is mutable.
        torch.nn.modules.rnn._VF = rnn_compat.VariableFunctionsShim()
        # Wrap all the rnns
        for x in rnn_compat.RNN_NAMES:
            wrap.new_rnn_cast(x.upper(), handle, verbose, post_fn=whitelist_post_fn)

    # Wrap all the RNN cells
    rnn_compat.whitelist_rnn_cells(handle, verbose)

    # 6) Place error+print message on banned functions.
    #    Or, if allow_banned, then cast to FP32.
    for fn, err_msg in functional_overrides.banned_funcs(config):
        if allow_banned:
            wrap.cached_cast(functional_overrides.MODULE, fn, utils.maybe_float,
                             handle, try_caching=True, verbose=verbose)
        else:
            wrap.err_if_any_half(functional_overrides.MODULE, fn, handle, err_msg)

    _DECORATOR_HANDLE = handle

    _amp_state.handle = handle

    return handle
