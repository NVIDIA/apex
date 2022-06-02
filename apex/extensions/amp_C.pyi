from typing import Sequence

import torch


def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    ...


def unflatten(flat: torch.Tensor, tensors: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    ...
