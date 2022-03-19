from typing import Optional, Sequence

import torch


def _get_autocast_dtypes() -> Sequence[torch.dtype]:
    if torch.cuda.is_bf16_supported():
        return [torch.half, torch.bfloat16]
    return [torch.half]


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
