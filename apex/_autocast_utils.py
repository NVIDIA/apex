from collections.abc import Sequence
from typing import Optional

import torch

__all__ = ["_cast_if_autocast_enabled"]


def _get_autocast_dtypes() -> Sequence[torch.dtype]:
    if torch.cuda.is_bf16_supported():
        return [torch.half, torch.bfloat16]
    return [torch.half]


def _get_current_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    if not torch.is_autocast_enabled():
        return torch.float or dtype
    else:
        return torch.get_autocast_gpu_dtype()


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        # torch.cuda.amp.autocast_mode._cast is deprecated since torch 2.4 and
        # scheduled for removal; torch.amp.autocast_mode._cast takes an
        # explicit device_type since torch 2.5.
        return torch.amp.autocast_mode._cast(args, "cuda", torch.get_autocast_gpu_dtype())
