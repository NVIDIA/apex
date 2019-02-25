from . import utils, wrap

import torch
_VF = torch._C._VariableFunctions
RNN_NAMES = ['rnn_relu', 'rnn_tanh', 'gru', 'lstm']

def _gen_VF_wrapper(name):
    def wrapper(*args, **kwargs):
        return getattr(_VF, name)(*args, **kwargs)
    return wrapper

# Some python magic to generate an object that has the rnn cell functions
# defined on it, all of which call into corresponding _VF version.
# Intended to patch torch.nn.modules.rnn._VF (aka, the ref named "_VF"
# imported at module scope within torch.nn.modules.rnn).  This should
# not affect third-party importers of _VF.py.
class VariableFunctionsShim(object):
    def __init__(self):
        for name in RNN_NAMES:
            for suffix in ['', '_cell']:
               fn_name = name + suffix
               setattr(self, fn_name, _gen_VF_wrapper(fn_name))

def has_old_rnns():
    try:
        torch.nn.backends.thnn.backend.LSTMCell
        return True
    except:
        return False

def whitelist_rnn_cells(handle, verbose):
    # Different module + function names in old/new RNN cases
    if has_old_rnns():
        fn_names = ['RNNReLUCell', 'RNNTanhCell', 'LSTMCell', 'GRUCell']
        mod = torch.nn.backends.thnn.backend
    else:
        fn_names = [x + '_cell' for x in RNN_NAMES]
        mod = torch.nn.modules.rnn._VF
        assert isinstance(mod, VariableFunctionsShim)

    # Insert casts on cell functions
    for fn in fn_names:
        wrap.cached_cast(mod, fn, utils.maybe_half, handle,
                         try_caching=True, verbose=verbose)

    if has_old_rnns():
        # Special handling of `backward` for fused gru / lstm:
        # The `backward` method calls Tensor.sum() (blacklist) internally,
        # and then the resulting grad_input has the wrong type.
        # TODO: where else is this a problem?
        for rnn_type in ['GRUFused', 'LSTMFused']:
            mod = getattr(torch.nn._functions.thnn.rnnFusedPointwise, rnn_type)
            wrap.disable_casts(mod, 'backward', handle)
