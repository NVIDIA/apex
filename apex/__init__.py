from typing import Optional
import logging
import os
import threading

# May help avoid undefined symbol errors https://pytorch.org/cppdocs/notes/faq.html#undefined-symbol-errors-from-pytorch-aten
import torch


_modules = []


if torch.distributed.is_available():
    from . import parallel
    _modules.append("parallel")

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


_modules.extend(["amp", "fp16_util", "optimizers", "normalization", "pyprof", "transformer"])


__all__ = [
    "get_transformer_logger", "set_logging_level",
] + _modules


_lock_for_logger: threading.Lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None


class RankInfoFormatter(logging.Formatter):

    def format(self, record):
        from apex.transformer.parallel_state import get_rank_info
        record.rank_info = get_rank_info()
        return super().format(record)


def _create_default_formatter() -> logging.Formatter:
    return RankInfoFormatter('%(asctime)s - %(name)s - %(levelname)s - (%(rank_info)s) - %(message)s')


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(__name__.split(".")[0])


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock_for_logger:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.setFormatter(_create_default_formatter())

        # Apply our default configuration to the library root logger.
        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.WARNING)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:

    global _default_handler

    with _lock_for_logger:
        if not _default_handler:
            return

        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)
        _default_handler = None


def get_transformer_logger(name: str) -> logging.Logger:
    name_wo_ext = os.path.splitext(name)[0]
    _configure_library_root_logger()
    return logging.getLogger(name_wo_ext)


def get_logging_level() -> int:
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_logging_level(verbosity: int) -> None:

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)
