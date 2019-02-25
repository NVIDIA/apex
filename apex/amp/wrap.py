from . import compat
from . import utils
from ._amp_state import _amp_state
from . import rnn_compat

import functools

import torch

def make_cast_wrapper(orig_fn, cast_fn, handle,
                      try_caching=False):
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        if not handle.is_active():
            return orig_fn(*args, **kwargs)

        if try_caching and handle.has_cache:
            args = list(args)
            for i in range(len(args)):
                if utils.should_cache(args[i]):
                    args[i] = utils.cached_cast(cast_fn, args[i], handle.cache)
            for k in kwargs:
                if utils.should_cache(kwargs[k]):
                    kwargs[k] = utils.cached_cast(cast_fn, kwargs[k], handle.cache)
        new_args = utils.casted_args(cast_fn,
                                     args,
                                     kwargs)
        return orig_fn(*new_args, **kwargs)
    return wrapper

def cached_cast(mod, fn, cast_fn, handle,
                try_caching=False, verbose=False):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(cast_fn, fn, verbose)
    wrapper = make_cast_wrapper(orig_fn, cast_fn, handle, try_caching)
    utils.set_func_save(handle, mod, fn, wrapper)

# `handle` arg is unused, but simplifies API to make `make_cast_wrapper`
# Annoyingly, make_promote_wrapper still uses the global handle.  Once everyone
# is on the new API and I am free to get rid of handle, I can clean this up.
def make_promote_wrapper(orig_fn, cast_fn, handle=None):
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        if not _amp_state.handle.is_active():
            return orig_fn(*args, **kwargs)

        types = utils.collect_fp_tensor_types(args, kwargs)

        if len(types) <= 1:
            return orig_fn(*args, **kwargs)
        elif len(types) == 2 and types == set(['HalfTensor', 'FloatTensor']):
            new_args = utils.casted_args(cast_fn,
                                         args,
                                         kwargs)
            return orig_fn(*new_args, **kwargs)
        else:
            raise NotImplementedError('Do not know how to handle ' +
                                      'these types to promote: {}'
                                      .format(types))
    return wrapper

def promote(mod, fn, handle, verbose=False):
    orig_fn = utils.get_func(mod, fn)
    maybe_float = utils.verbosify(utils.maybe_float, fn, verbose)
    wrapper = make_promote_wrapper(orig_fn, maybe_float)
    utils.set_func_save(handle, mod, fn, wrapper)

def sequence_promote(mod, fn, handle, verbose=False):
    orig_fn = utils.get_func(mod, fn)
    maybe_float = utils.verbosify(utils.maybe_float, fn, verbose)
    @functools.wraps(orig_fn)
    def wrapper(seq, *args, **kwargs):
        if not _amp_state.handle.is_active():
            return orig_fn(seq, *args, **kwargs)

        types = set([utils.type_string(x) for x in seq])
        if len(types) <= 1:
            return orig_fn(seq, *args, **kwargs)
        elif types == set(['HalfTensor', 'FloatTensor']):
            cast_seq = utils.casted_args(maybe_float,
                                         seq, {})
            return orig_fn(cast_seq, *args, **kwargs)
        else:
            # TODO: other mixed-type cases aren't due to amp.
            #       Just pass through?
            return orig_fn(seq, *args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)

def promote_match_arg0(mod, fn, handle, verbose=False):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    def wrapper(arg0, *args, **kwargs):
        assert compat.is_tensor_like(arg0)
        if not _amp_state.handle.is_active():
            return orig_fn(arg0, *args, **kwargs)

        if utils.type_string(arg0) == 'HalfTensor':
            cast_fn = utils.maybe_half
        elif utils.type_string(arg0) == 'FloatTensor':
            cast_fn = utils.maybe_float
        else:
            return orig_fn(arg0, *args, **kwargs)
        cast_fn = utils.verbosify(cast_fn, fn, verbose)
        new_args = utils.casted_args(cast_fn, args, kwargs)
        return orig_fn(arg0, *new_args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)

def err_if_any_half(mod, fn, handle, custom_err_msg=None):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        types = utils.collect_fp_tensor_types(args, kwargs)
        if 'HalfTensor' in types:
            if custom_err_msg:
                raise NotImplementedError(custom_err_msg)
            else:
                raise NotImplementedError('Cannot call in-place function ' +
                                          '{} with fp16 arguments.'.format(fn))
        else:
            return orig_fn(*args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)

def err_if_arg0_half(mod, fn, handle, verbose=False):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    def wrapper(arg0, *args, **kwargs):
        assert compat.is_tensor_like(arg0)
        if utils.type_string(arg0) == 'HalfTensor':
            raise NotImplementedError('Cannot call in-place method ' +
                                      '{} on fp16 Tensors.'.format(fn))
        else:
            cast_fn = utils.verbosify(utils.maybe_float, fn, verbose)
            new_args = utils.casted_args(cast_fn, args, kwargs)
            return orig_fn(arg0, *new_args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)

# Current RNN approach:
# - Wrap top-level `RNN` function in thnn backend
# - Will call into either CudnnRNN or AutogradRNN
#  - Each of these are factory functions that return a per-iter
#    `forward` function
# - We interpose on the factory function to:
#   1) Interpose on the actual forward function and put in casts
#   2) Insert an fp16 `flat_weight` if necessary
def rnn_cast(backend, fn, handle, verbose=False):
    orig_rnn = utils.get_func(backend, fn)
    @functools.wraps(orig_rnn)
    def rnn_wrapper(*args, **kwargs):
        flat_weight = kwargs.get('flat_weight')
        if flat_weight is not None:
            # We replace `flat_weight` with an uninitialized fp16
            # Tensor. The "actual" weight tensors (provided in `forward`),
            # will then be set up as ptrs into the buffer and have the
            # corresponding fp32 values copied in.
            # We need to call `copy` on the "actual" weights so that the
            # autograd graph correctly backprops from the wgrads computed
            # inside cuDNN (on fp16 weights) into the fp32 weights.
            assert utils.type_string(flat_weight) == 'FloatTensor'
            if compat.tensor_is_float_tensor() or compat.tensor_is_variable():
                # Pre-0.4. A little slower, since it zeros out memory.
                flat_weight_fp16 = flat_weight.new().half().resize_(flat_weight.shape)
            else:
                flat_weight_fp16 = torch.empty_like(flat_weight,
                                                    dtype=torch.float16)
            kwargs['flat_weight'] = flat_weight_fp16
        else:
            flat_weight_fp16 = None

        forward = orig_rnn(*args, **kwargs)
        @functools.wraps(forward)
        def fwd_wrapper(*fargs, **fkwargs):
            assert len(fargs) == 3 or len(fargs) == 4
            inputs, weights, hiddens = fargs[:3]
            assert utils.is_fp_tensor(inputs)
            assert isinstance(weights, list)
            cast_fn = utils.verbosify(utils.maybe_half,
                                      fn,
                                      verbose)
            new_args = []

            # 0) Inputs
            new_args.append(cast_fn(inputs))

            # 1) Weights
            if flat_weight_fp16 is not None:
                fp16_weights = utils.synthesize_flattened_rnn_weights(
                    weights, flat_weight_fp16, fn, verbose)
            else:
                fp16_weights = [[cast_fn(w) for w in layer]
                                for layer in weights]
            new_args.append(fp16_weights)

            # 2) Inputs: either a tuple (for LSTM) or single tensor
            if isinstance(hiddens, tuple):
                new_args.append(tuple(cast_fn(x) for x in hiddens))
            elif utils.is_fp_tensor(hiddens):
                new_args.append(cast_fn(hiddens))
            else:
                # Hiddens can, in principle, be `None` -- pass through
                new_args.append(hiddens)

            # 3) Batch sizes (0.4 or later only)
            if len(fargs) == 4:
                new_args.append(fargs[3])

            return forward(*new_args, **fkwargs)
        return fwd_wrapper
    utils.set_func_save(handle, backend, fn, rnn_wrapper)

def new_rnn_cast(fn, handle, verbose=False):
    # Forward+backward compatibility around https://github.com/pytorch/pytorch/pull/15744
    # For rnn backend calls that route through _rnn_impls, we must patch the ref
    # that _rnn_impls stashed.  For rnn backend calls that directly invoke
    # _VF.<backend>, e.g. _VF.lstm, we can patch onto VariableFunctionsShim,
    # which in turn has patched the ref named "_VF" in torch.nn.modules.rnn.
    if utils.has_func(torch.nn.modules.rnn._rnn_impls, fn):
        mod = torch.nn.modules.rnn._rnn_impls
    else:
        mod = torch.nn.modules.rnn._VF
        assert isinstance(mod, rnn_compat.VariableFunctionsShim)
        fn = fn.lower()
    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(utils.maybe_half, fn, verbose)
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        # Exact call signature from modules/rnn.py
        assert len(args) == 9
        assert len(kwargs) == 0

        if not _amp_state.handle.is_active():
            return orig_fn(*args, **kwargs)

        if isinstance(args[6], bool):
            params_idx = 2 # Not PackedSequence case
        else:
            params_idx = 3 # PackedSequence case

        new_args = []
        for i, arg in enumerate(args):
            if i == params_idx:
                num_params = sum([x.numel() for x in arg])
                fp16_weight_buf = args[0].new_empty((num_params,),
                                                    dtype=torch.half)
                casted_weights = utils.new_synthesize_flattened_rnn_weights(
                    arg, fp16_weight_buf, fn, verbose)
                new_args.append(casted_weights)
            elif utils.is_fp_tensor(arg):
                new_args.append(cast_fn(arg))
            else:
                new_args.append(arg)

        return orig_fn(*new_args)
    utils.set_func_save(handle, mod, fn, wrapper)

def disable_casts(mod, fn, handle):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        with handle._disable_casts():
            return orig_fn(*args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)
