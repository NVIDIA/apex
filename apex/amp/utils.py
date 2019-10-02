from . import compat

import functools
import itertools

import torch

def is_cuda_enabled():
    return torch.version.cuda is not None

def get_cuda_version():
    return tuple(int(x) for x in torch.version.cuda.split('.'))

def is_fp_tensor(x):
    if is_nested(x):
        # Fast-fail version of all(is_fp_tensor)
        for y in x:
            if not is_fp_tensor(y):
                return False
        return True
    return compat.is_tensor_like(x) and compat.is_floating_point(x)

def is_nested(x):
    return isinstance(x, tuple) or isinstance(x, list)

def should_cache(x):
    if is_nested(x):
        # Fast-fail version of all(should_cache)
        for y in x:
            if not should_cache(y):
                return False
        return True
    return isinstance(x, torch.nn.parameter.Parameter) and \
        type_string(x) == 'FloatTensor'

def collect_fp_tensor_types(args, kwargs):
    def collect_types(x, types):
        if is_nested(x):
            for y in x:
                collect_types(y, types)
        else:
            types.add(type_string(x))

    all_args = itertools.chain(args, kwargs.values())
    types = set()
    for x in all_args:
        if is_fp_tensor(x):
            collect_types(x, types)
    return types

def type_string(x):
    return x.type().split('.')[-1]

def maybe_half(x, name='', verbose=False):
    if is_nested(x):
        return type(x)([maybe_half(y) for y in x])

    if not x.is_cuda or type_string(x) == 'HalfTensor':
        return x
    else:
        if verbose:
            print('Float->Half ({})'.format(name))
        return x.half()

def maybe_float(x, name='', verbose=False):
    if is_nested(x):
        return type(x)([maybe_float(y) for y in x])

    if not x.is_cuda or type_string(x) == 'FloatTensor':
        return x
    else:
        if verbose:
            print('Half->Float ({})'.format(name))
        return x.float()

# NB: returneds casted `args`, mutates `kwargs` in-place
def casted_args(cast_fn, args, kwargs):
    new_args = []
    for x in args:
        if is_fp_tensor(x):
            new_args.append(cast_fn(x))
        else:
            new_args.append(x)
    for k in kwargs:
        val = kwargs[k]
        if is_fp_tensor(val):
            kwargs[k] = cast_fn(val)
    return new_args

def cached_cast(cast_fn, x, cache):
    if is_nested(x):
        return type(x)([cached_cast(y) for y in x])
    if x in cache:
        cached_x = cache[x]
        if x.requires_grad and cached_x.requires_grad:
            # Make sure x is actually cached_x's autograd parent.
            if cached_x.grad_fn.next_functions[1][0].variable is not x:
                raise RuntimeError("x and cache[x] both require grad, but x is not "
                                   "cache[x]'s parent.  This is likely an error.")
        # During eval, it's possible to end up caching casted weights with
        # requires_grad=False.  On the next training iter, if cached_x is found
        # and reused from the cache, it will not actually have x as its parent.
        # Therefore, we choose to invalidate the cache (and force refreshing the cast)
        # if x.requires_grad and cached_x.requires_grad do not match.
        #
        # During eval (i.e. running under with torch.no_grad()) the invalidation
        # check would cause the cached value to be dropped every time, because
        # cached_x would always be created with requires_grad=False, while x would
        # still have requires_grad=True.  This would render the cache effectively
        # useless during eval.  Therefore, if we are running under the no_grad()
        # context manager (torch.is_grad_enabled=False) we elide the invalidation
        # check, and use the cached value even though its requires_grad flag doesn't
        # match.  During eval, we don't care that there's no autograd-graph
        # connection between x and cached_x.
        if torch.is_grad_enabled() and x.requires_grad != cached_x.requires_grad:
            del cache[x]
        else:
            return cached_x

    casted_x = cast_fn(x)
    cache[x] = casted_x
    return casted_x

def verbosify(cast_fn, fn_name, verbose):
    if verbose:
        return functools.partial(cast_fn, name=fn_name, verbose=verbose)
    else:
        return cast_fn

def as_inplace(fns):
    for x in fns:
        yield x + '_'

def has_func(mod, fn):
    if isinstance(mod, dict):
        return fn in mod
    else:
        return hasattr(mod, fn)

def get_func(mod, fn):
    if isinstance(mod, dict):
        return mod[fn]
    else:
        return getattr(mod, fn)

def set_func(mod, fn, new_fn):
    if isinstance(mod, dict):
        mod[fn] = new_fn
    else:
        setattr(mod, fn, new_fn)

def set_func_save(handle, mod, fn, new_fn):
    cur_fn = get_func(mod, fn)
    handle._save_func(mod, fn, cur_fn)
    set_func(mod, fn, new_fn)

# A couple problems get solved here:
# - The flat_weight buffer is disconnected from autograd graph,
#   so the fp16 weights need to be derived from the input weights
#   to this forward call, not the flat buffer.
# - The ordering of weights in the flat buffer is...idiosyncratic.
# First problem is solved with combination of set_ (to set up
# correct storage) and copy_ (so the fp16 weight derives from the
# fp32 one in autograd.
# Second is solved by doing ptr arithmetic on the fp32 weights
# to derive the correct offset.
#
# TODO: maybe this should actually use
# `torch._cudnn_rnn_flatten_weight`? But then I need to call
# on first iter and cache the right offsets. Ugh.
def synthesize_flattened_rnn_weights(fp32_weights,
                                     fp16_flat_tensor,
                                     rnn_fn='',
                                     verbose=False):
    fp16_weights = []
    fp32_base_ptr = fp32_weights[0][0].data_ptr()
    for layer_weights in fp32_weights:
        fp16_layer_weights = []
        for w_fp32 in layer_weights:
            w_fp16 = w_fp32.new().half()
            offset = (w_fp32.data_ptr() - fp32_base_ptr) // w_fp32.element_size()
            w_fp16.set_(fp16_flat_tensor.storage(),
                        offset,
                        w_fp32.shape)
            w_fp16.copy_(w_fp32)
            if verbose:
                print('Float->Half ({})'.format(rnn_fn))
            fp16_layer_weights.append(w_fp16)
        fp16_weights.append(fp16_layer_weights)
    return fp16_weights

# Roughly same as above, just the `fp32_weights` aren't nested.
# Code kept separate for readability.
def new_synthesize_flattened_rnn_weights(fp32_weights,
                                         fp16_flat_tensor,
                                         rnn_fn='',
                                         verbose=False):
    fp16_weights = []
    fp32_base_ptr = fp32_weights[0].data_ptr()
    for w_fp32 in fp32_weights:
        w_fp16 = w_fp32.new().half()
        offset = (w_fp32.data_ptr() - fp32_base_ptr) // w_fp32.element_size()
        w_fp16.set_(fp16_flat_tensor.storage(),
                    offset,
                    w_fp32.shape)
        w_fp16.copy_(w_fp32)
        if verbose:
            print('Float->Half ({})'.format(rnn_fn))
        fp16_weights.append(w_fp16)
    return fp16_weights
