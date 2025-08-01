"""Graph scheduler package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch._inductor
from torch._dynamo import list_backends
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx_inner

from .backend import get_backend

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from torch._ops import OpOverload

__all__ = ["get_backend", "set_default_backend"]

# Register custom operators
torch.ops.import_module("apex.contrib.torchsched.ops")


# Register torch-sched backend
# Same API as torch._inductor.compile_fx
@register_backend
def torchsched(
    model_: torch.fx.GraphModule,
    example_inputs_: list[torch.Tensor],
    inner_compile: Callable[..., Any] = compile_fx_inner,
    config_patches: dict[str, Any] | None = None,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
) -> Callable:
    backend = get_backend(backend="torchsched", scheme="dwb")
    return backend(model_, example_inputs_, inner_compile, config_patches, decompositions)


_SUPPORTED_BACKENDS = list_backends()
_DEFAULT_BACKEND = "inductor"
__torch_compile__ = torch.compile


def set_default_backend(backend: str) -> None:
    """
    Set the default backend for torch.compile.

    Parameters:
        backend (str): The backend to use as the default for torch.compile.
    """
    global _DEFAULT_BACKEND
    assert backend in _SUPPORTED_BACKENDS, f"Unknown backend {backend}"
    _DEFAULT_BACKEND = backend


def torchsched_compile(
    *args: object,
    backend: str | Callable | None = None,
    **kwargs: object,
) -> object:
    """
    Wrap around the original torch.compile to support default backend.

    Parameters:
        *args (object): Positional arguments for torch.compile.
        backend (Union[str, Callable, None]): The backend to use.
            If None, the default backend is used.
        **kwargs (object): Additional keyword arguments for torch.compile.

    Returns:
        object: Compiler or compiled model.
    """
    if backend is None:
        backend = _DEFAULT_BACKEND
    return __torch_compile__(*args, backend=backend, **kwargs)


# Monkey patch torch.compile to set default backend
torch.compile = torchsched_compile
