
from apex import amp
from apex.fp16_utils import FP16_Optimizer

# TODO: what we want is for this to construct a class object that exposes (roughly)
# the same API as this module-level API (thereby ensuring there's only one of them,
# since modules are imported only once per interpreter).
_amp_handle = None

def enable_automatic_conversion():
    global _amp_handle
    if _amp_handle is not None:
        raise RuntimeError('Cannot call `enable_automatic_conversion` more than once.')
    _amp_handle = amp.init(enabled=True)

def wrap_optimizer(optimizer):
    
