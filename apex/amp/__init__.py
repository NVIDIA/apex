from .amp import init, half_function, float_function, promote_function,\
    register_half_function, register_float_function, register_promote_function
from .handle import scale_loss, disable_casts
from .frontend import initialize
from ._amp_state import master_params, _amp_state
