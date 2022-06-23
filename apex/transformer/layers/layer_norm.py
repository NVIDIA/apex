# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# NOTE(mkozuki): This file defines two LayerNorm that are compatible with Megatron-LM.
# while avoiding introducing the breaking change of `"sequence_parallel_enabled"` attribute into apex.normalization.FusedLayerNorm
# and apex.contrib.layer_norm.FastLayerNorm.
import warnings

import torch

from apex.normalization import FusedLayerNorm as OrigFusedLayerNorm
from apex.normalization import MixedFusedLayerNorm as OrigMixedFusedLayerNorm
try:
    from apex.contrib.layer_norm import FastLayerNorm as OrigFastLayerNorm
except ImportError:
    HAS_FAST_LAYER_NORM = False
else:
    HAS_FAST_LAYER_NORM = True


__all__ = [
    "FusedLayerNorm",
    "FastLayerNorm",
    "MixedFusedLayerNorm",
]


def _set_sequence_parallel_enabled(
    param: torch.Tensor,
    sequence_parallel_enabled: bool,
) -> None:
    setattr(param, "sequence_parallel_enabled", sequence_parallel_enabled)


class FusedLayerNorm(OrigFusedLayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.elementwise_affine:
            _set_sequence_parallel_enabled(self.weight, self.sequence_parallel_enabled)
            _set_sequence_parallel_enabled(self.bias, self.sequence_parallel_enabled)


# note: MixedFusedLayerNorm is no different from FusedLayerNorm if it's used in `torch.cuda.amp`.
class MixedFusedLayerNorm(OrigMixedFusedLayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        self.sequence_parallel_enabled = kwargs.get("sequence_parallel_enabled", False)
        super().__init__(normalized_shape=normalized_shape, eps=eps, **kwargs)
        if self.sequence_parallel_enabled:
            _set_sequence_parallel_enabled(self.weight, self.sequence_parallel_enabled)
            _set_sequence_parallel_enabled(self.bias, self.sequence_parallel_enabled)


if HAS_FAST_LAYER_NORM:
    class FastLayerNorm(OrigFastLayerNorm):
        def __init__(
            self,
            hidden_size,
            eps: float = 1e-5,
            *,
            sequence_parallel_enabled: bool = False,
        ):
            super().__init__(
                hidden_size=hidden_size,
                eps=eps
            )
            self.sequence_parallel_enabled = sequence_parallel_enabled
            _set_sequence_parallel_enabled(self.weight, self.sequence_parallel_enabled)
            _set_sequence_parallel_enabled(self.bias, self.sequence_parallel_enabled)
else:
    class FastLayerNorm(FusedLayerNorm):
        def __init__(
            self,
            hidden_size,
            eps: float = 1e-5,
            *,
            sequence_parallel_enabled: bool = False,
        ):
            warnings.warn("`apex.contrib.layer_norm.FastLayerNorm` isn't available thus falling back to `apex.normalization.FusedLayerNorm`")
            super().__init__(
                normalized_shape=hidden_size,
                eps=eps,
                elementwise_affine=True,
                sequence_parallel_enabled=sequence_parallel_enabled,
            )
