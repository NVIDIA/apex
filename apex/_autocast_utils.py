from typing import Optional

import torch


def _get_current_dtype(dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if not torch.is_autocast_enabled():
        return torch.float or dtype
    else:
        return torch.get_autocast_gpu_dtype()


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
