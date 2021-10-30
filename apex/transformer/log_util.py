from typing import Optional
import logging
import os
import threading


def get_transformer_logger(name: str) -> logging.Logger:
    name_wo_ext = os.path.splitext(name)[0]
    return logging.getLogger(name_wo_ext)


def set_logging_level(verbosity) -> None:
    """Change logging severity.

    Args:
        verbosity
    """
    from apex import _library_root_logger
    _library_root_logger.setLevel(verbosity)
