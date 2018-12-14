# from . import RNN
# from . import reparameterization
from . import fp16_utils
from . import parallel
from . import amp
try:
    from . import optimizers
except ImportError:
    # An attempt to fix https://github.com/NVIDIA/apex/issues/97.  I'm not sure why 97 is even
    # happening because Python modules should only be imported once, even if import is called
    # multiple times.
    try:
        _ = warned_optimizers
    except NameError:
        print("Warning:  apex was installed without --cuda_ext.  FusedAdam will be unavailable.")
        warned_optimizers = True
try:
    from . import normalization
except ImportError:
    try:
        _ = warned_normalization
    except NameError:
        print("Warning:  apex was installed without --cuda_ext.  FusedLayerNorm will be unavailable.")
        warned_normalization = True

