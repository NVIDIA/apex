import logging

# May help avoid undefined symbol errors https://pytorch.org/cppdocs/notes/faq.html#undefined-symbol-errors-from-pytorch-aten
import torch


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
from . import transformer


# Logging utilities for apex.transformer module
class RankInfoFormatter(logging.Formatter):

    def format(self, record):
        from apex.transformer.parallel_state import get_rank_info
        record.rank_info = get_rank_info()
        return super().format(record)


_library_root_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(RankInfoFormatter("%(asctime)s - PID:%(process)d - rank:%(rank_info)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s", "%y-%m-%d %H:%M:%S"))
_library_root_logger.addHandler(handler)
_library_root_logger.propagate = False
