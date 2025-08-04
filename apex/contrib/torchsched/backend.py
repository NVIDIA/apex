"""Graph scheduler backend."""

from __future__ import annotations

import functools
from copy import copy
from typing import TYPE_CHECKING
from typing import ParamSpec
from typing import TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import NotImplementedType

import torch
from torch import Tensor
from torch import _TorchCompileInductorWrapper
from torch._dynamo import lookup_backend
from torch._inductor.compile_fx import compile_fx
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table

import apex.contrib.torchsched.config as config
from apex.contrib.torchsched.inductor import patch_graph_lowering
from apex.contrib.torchsched.passes import pre_grad_custom_pass

aten = torch.ops.aten
prims = torch.ops.prims

__all__ = ["get_backend"]


P = ParamSpec("P")
R = TypeVar("R")


def enable_multi_stream_scheduling(compile_fn: Callable[P, R]) -> Callable[P, R]:
    assert callable(compile_fn)

    @functools.wraps(compile_fn)
    def _compile_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        patch_graph_lowering(patch=True)
        compile_results = compile_fn(*args, **kwargs)
        patch_graph_lowering(patch=False)
        return compile_results

    return _compile_wrapper


# Refer: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/decomposition.py#L213
def convolution_backward_decomp_dwb(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias_sizes: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
    output_mask: tuple[bool, bool, bool],
) -> tuple[Tensor, Tensor, Tensor] | NotImplementedType:
    """Decomposite convolution bprop using the dgrad/wgrad/bgrad scheme.

    Args:
        grad_output (Tensor): The gradient w.r.t output.
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.
        bias_sizes (Tuple[int, ...]): The sizes of the bias tensor.
        stride (Tuple[int, ...]): The stride of the convolution.
        padding (Tuple[int, ...]): The padding of the convolution.
        dilation (Tuple[int, ...]): The dilation of the convolution.
        transposed (bool): Whether the convolution is transposed.
        output_padding (Tuple[int, ...]): The output padding for the transposed convolution.
        groups (int): The number of groups for the convolution.
        output_mask (Tuple[bool, bool, bool]): A mask indicating which gradients to compute.

    Returns:
        Union[Tuple[Tensor, Tensor, Tensor], NotImplemented]: A tuple containing the
            gradients of the input, weight, and bias, or NotImplemented if the
            conditions are not met.
    """
    if not output_mask[2] or grad_output.device.type != "cuda":
        return NotImplemented
    grad_inp, _, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], False, False],
    )
    _, grad_weight, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [False, output_mask[1], False],
    )
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    return (grad_inp, grad_weight, grad_bias)


def convolution_backward_decomp_wbd(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias_sizes: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
    output_mask: tuple[bool, bool, bool],
) -> tuple[Tensor, Tensor, Tensor] | NotImplementedType:
    """Decomposite convolution bprop using the wgrad/bgrad/dgrad scheme.

    Args:
        grad_output (Tensor): The gradient w.r.t output.
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.
        bias_sizes (Tuple[int, ...]): The sizes of the bias tensor.
        stride (Tuple[int, ...]): The stride of the convolution.
        padding (Tuple[int, ...]): The padding of the convolution.
        dilation (Tuple[int, ...]): The dilation of the convolution.
        transposed (bool): Whether the convolution is transposed.
        output_padding (Tuple[int, ...]): The output padding for the transposed convolution.
        groups (int): The number of groups for the convolution.
        output_mask (Tuple[bool, bool, bool]): A mask indicating which gradients to compute.

    Returns:
        Union[Tuple[Tensor, Tensor, Tensor], NotImplemented]: A tuple containing the
            gradients of the input, weight, and bias, or NotImplemented if the
            conditions are not met.
    """
    if not output_mask[2] or grad_output.device.type != "cuda":
        return NotImplemented
    _, grad_weight, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [False, output_mask[1], False],
    )
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, _, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], False, False],
    )
    return (grad_inp, grad_weight, grad_bias)


class DecompositionsWrapper(_TorchCompileInductorWrapper):
    """A wrapper class for handling decompositions in model compilation.

    This class extends the `_TorchCompileInductorWrapper` to include additional
    decompositions for model compilation.

    Args:
        mode (str): The mode for the wrapper.
        options (Optional[Dict]): Additional options for the wrapper.
        dynamic (bool): Whether the wrapper is dynamic.
        decompositions (Dict): A dictionary of decompositions to use.

    Attributes:
        decompositions (Dict): The decompositions used by the wrapper.
    """

    def __init__(
        self,
        mode: str,
        options: dict | None,
        dynamic: bool,
        decompositions: dict,
    ) -> None:
        """Initialize the DecompositionsWrapper."""
        super().__init__(mode, options, dynamic)
        self.decompositions = decompositions
        # Force skip the type checking in self.apply_options() since default values are None type.
        self.config.update(
            {
                "pre_grad_custom_pass": (
                    pre_grad_custom_pass if config.enable_pre_grad_pass else None
                ),
            },
        )

    def __eq__(self, rhs: object) -> bool:
        """Check equality with another DecompositionsWrapper.

        Args:
            rhs (object): The other object to compare with.

        Returns:
            bool: True if the wrappers are equal, False otherwise.
        """
        eq = (
            isinstance(rhs, DecompositionsWrapper)
            and super().__eq__(rhs)
            and rhs.decompositions == self.decompositions
        )
        return eq

    def __call__(
        self,
        model_: torch.nn.Module,
        inputs_: list,
        *args: object,
        **kwargs: object,
    ) -> Callable:
        """Compiles the model with the given inputs and decompositions.

        Args:
            model_ (torch.nn.Module): The model to compile.
            inputs_ (list): The inputs to the model.
            args (object): Positional argument.
            kwargs (object): Keyword argument.

        Returns:
            Callable: The compiled model.
        """
        # Modifications to compilation process should be isolated between each compilations.
        decompositions = copy(select_decomp_table())
        decompositions.update(self.decompositions)
        return compile_fx(
            model_,
            inputs_,
            inner_compile=enable_multi_stream_scheduling(compile_fx_inner),
            config_patches=self.config,
            decompositions=decompositions,
        )


def get_backend(
    backend: str = "torch",
    scheme: str = "dwb",
) -> Callable | DecompositionsWrapper:
    """Get the graph scheduler backend for model compilation.

    This function returns the appropriate backend for model compilation based on
    the specified parameters.

    Args:
        backend (str, optional): The backend to use. Defaults to "torch".
        scheme (str, optional): The decomposition scheme to use. Defaults to "dwb".

    Returns:
        Union[Callable, DecompositionsWrapper]: The backend for model compilation.

    Raises:
        Exception: If an unknown scheme is specified.
    """
    if backend not in ("torch", "torchsched"):
        raise ValueError(f"Unknown compilation {backend=}")
    if scheme not in ("dwb", "wbd"):
        raise ValueError(f"Invalid {scheme=}, use scheme=dwb or wbd instead")

    if backend == "torch":
        return lookup_backend("inductor")

    # [NOTE] Disable buffer reuse and inplace buffers to avoid inter-stream conflicts.
    #
    # In PyTorch Inductor, the safety of buffer reuse and in-place buffer update is ensured by the
    # program's single-stream, serial execution. That is, if op2 is launched only after op1 has
    # completed execution, then these cases are safe:
    #
    #   Case 1: Safe to reuse buffer `workspace1` as `op2`'s workspace.
    #
    #         op1   ->   op2              op1   ->   op2
    #          ↕          ↕       ⇒        ↕          ↑
    #     workspace1 workspace2       workspace1 ←----┘
    #
    #   Case 2: Safe to inpalace `op1`'s output to `buf1` then send to `op2` as input.
    #
    #     buf1 -> op1 -> buf2 -> op2  ⇒  buf1 ↔	op1
    #                                     └-------> op2
    #
    # However, if operators are dispatched to distinct CUDA Streams and execute in parallel, above
    # cases are not safe any more:
    #
    #   Counter example 1: Case 1 is not safe if op1 and op2 are in parallel.
    #
    #        op1
    #         ↕
    #     workspace1 (Buffer modified concurrently by op1 and op2.)
    #         ↕
    #        op2
    #
    #   Counter example 2: Case 2 is not safe if op1 and op2 are in parallel.
    #
    #     buf1 <-->	op1
    #      └------> op2 (Op2 could read op1's input data.)
    #
    # Thus currently we disable both buffer reuse and inplace buffer update to ensure multi-stream
    # correctness.
    #
    # TODO(@davidli): Add cross-stream dependency to Inductor scheduling's dependency system so we
    # can safely reuse and inplace update buffers even in multi-stream scenario.

    if scheme == "dwb":
        return DecompositionsWrapper(
            mode="default",
            options={"allow_buffer_reuse": False, "inplace_buffers": False},
            dynamic=False,
            decompositions={
                aten.convolution_backward.default: convolution_backward_decomp_dwb,
            },
        )
    elif scheme == "wbd":
        return DecompositionsWrapper(
            mode="default",
            options={"allow_buffer_reuse": False, "inplace_buffers": False},
            dynamic=False,
            decompositions={
                aten.convolution_backward.default: convolution_backward_decomp_wbd,
            },
        )
    else:
        # To please mypy
        raise ValueError(f"Invalid {scheme=}, use scheme=dwb or wbd instead")
