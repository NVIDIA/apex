# © 2023 NVIDIA CORPORATION & AFFILIATES

import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
from typing import BinaryIO, Union

import torch
from triton.runtime.autotuner import Autotuner, Config, Heuristics
from triton.runtime.jit import JITFunction

from apex.contrib.openfold_triton._layer_norm_backward_kernels import (
    _layer_norm_backward_dw_db_partial,
    _layer_norm_backward_dw_db_partial_strided,
    _layer_norm_backward_dx,
    _layer_norm_backward_dx_strided,
)
from apex.contrib.openfold_triton._layer_norm_forward_kernels import (
    _layer_norm_forward,
    _layer_norm_forward_strided,
)
from apex.contrib.openfold_triton.layer_norm import LayerNormSmallShapeOptImpl
from apex.contrib.openfold_triton.mha import (
    AttnBiasJIT,
    AttnNoBiasJIT,
    AttnTri,
    CanSchTriMHA,
)

__all__ = (
    "LayerNormSmallShapeOptImpl",
    "sync_triton_auto_tune_cache_across_gpus",
    "CanSchTriMHA",
    "AttnTri",
    "AttnBiasJIT",
    "AttnNoBiasJIT",
)


def _get_tuneable_triton_func_name(f: Union[Autotuner, Heuristics, JITFunction]) -> str:
    if isinstance(f, JITFunction):
        return f.__name__
    else:
        return _get_tuneable_triton_func_name(f.fn)


_tuneable_triton_kernels = OrderedDict(
    (_get_tuneable_triton_func_name(func), func)
    for func in (
        _layer_norm_backward_dw_db_partial,
        _layer_norm_backward_dw_db_partial_strided,
        _layer_norm_backward_dx,
        _layer_norm_backward_dx_strided,
        _layer_norm_forward,
        _layer_norm_forward_strided,
    )
)


def _save_triton_auto_tune_cache(strict: bool = True, verbose: bool = False) -> BytesIO:
    caches = OrderedDict()
    for func_name, func in _tuneable_triton_kernels.items():
        if len(func.cache) < 1:
            msg = f"Triton JIT kernel {func_name} didn't have tuning cache"
            if strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
        else:
            caches[func_name] = [
                (keys, vals.all_kwargs())
                for keys, vals in zip(func.cache.keys(), func.cache.values())
            ]
    f = BytesIO(json.dumps(caches).encode("utf-8"))
    if verbose:
        print(f"Triton kernel auto-tuning caches written to {f}")
    return f


def _load_triton_auto_tune_cache(f: BinaryIO, strict: bool = True, verbose: bool = False) -> None:
    caches = json.load(f)
    if strict:
        loaded_func_name = set(caches.keys())
        tuneable_func_name = set(_tuneable_triton_kernels.keys())
        if loaded_func_name != tuneable_func_name:
            raise ValueError(
                f"Tuneable Triton kernels don't match with provided auto-tuning cache file {f}\n"
                f"Missing kernel caches: {tuneable_func_name - loaded_func_name}\n"
                f"Unexpected kernel caches: {loaded_func_name - tuneable_func_name}"
            )
    for func_name, func_cache in caches.items():
        if func_name not in _tuneable_triton_kernels:
            raise ValueError(f"{func_name} from {f} doesn't match any tuneable Triton kernels")
        for key, val in func_cache:
            _tuneable_triton_kernels[func_name].cache[tuple(key)] = Config(val)
    if verbose:
        print(f"Triton kernel auto-tuning caches loaded from {f}")


def sync_triton_auto_tune_cache_across_gpus(strict: bool = True, verbose: bool = False) -> None:
    if not torch.distributed.is_initialized():
        return
    if torch.distributed.get_rank() == 0:
        print("Broadcasting Triton auto-tuning cache from rank 0 to other ranks...")
        cache = _save_triton_auto_tune_cache(strict=strict, verbose=verbose)
        cache.seek(0)
        cache_list = [
            cache,
        ]
    else:
        print(
            f"Rank {torch.distributed.get_rank()} is waiting for Triton auto-tuning cache from rank 0..."
        )
        cache_list = [
            None,
        ]
    torch.distributed.broadcast_object_list(cache_list)
    _load_triton_auto_tune_cache(cache_list[0], strict=strict, verbose=verbose)
    print("Succeed!")
