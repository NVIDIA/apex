# May help avoid undefined symbol errors https://pytorch.org/cppdocs/notes/faq.html#undefined-symbol-errors-from-pytorch-aten
import torch
import warnings

if torch.distributed.is_available():
    from . import parallel

from . import amp
from . import fp16_utils

# For optimizers and normalization there is no Python fallback.
# Absence of cuda backend is a hard error.
# I would like the errors from importing fused_adam_cuda or fused_layer_norm_cuda
# to be triggered lazily, because if someone has installed with --cpp_ext and --cuda_ext
# so they expect those backends to be available, but for some reason they actually aren't
# available (for example because they built improperly in a way that isn't revealed until
# load time) the error message is timely and visible.
from . import optimizers
from . import normalization
from . import pyprof
