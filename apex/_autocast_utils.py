from typing import Optional

import torch


_has_get_autocast_gpu_dtype_available = hasattr(torch, "get_autocast_gpu_dtype")


def _get_current_dtype(dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if not torch.is_autocast_enabled():
        return torch.float or dtype
    else:
        if not _has_get_autocast_gpu_dtype_available:
            return torch.half
        return torch.get_autocast_gpu_dtype()


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
