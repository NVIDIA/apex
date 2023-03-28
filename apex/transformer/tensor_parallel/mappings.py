# coding=utf-8
# Copyright (c) 2021-22, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from apex.transformer.parallel_state import get_tensor_model_parallel_group
from apex.transformer.parallel_state import get_tensor_model_parallel_world_size
from apex.transformer.parallel_state import get_tensor_model_parallel_rank
from apex.transformer.tensor_parallel.utils import split_tensor_along_last_dim

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 4 lines are for backward comparability with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base

def _reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its first dimension and keep the corresponding slice."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU for tensor model parallel.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size(0)
    assert dim_size % world_size == 0
    local_dim_size = dim_size // world_size
    dim_offset = get_tensor_model_parallel_rank() * local_dim_size
    output = input_[dim_offset:dim_offset + local_dim_size].contiguous()
    return output


def _gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatenate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(
        tensor_list, input_, group=get_tensor_model_parallel_group()
    )

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatenate along the first dimension."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    shape = list(input_.shape)
    shape[0] *= world_size

    output = torch.empty(shape, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output,
        input_.contiguous(),
        group=get_tensor_model_parallel_group()
        )
    return output


def _reduce_scatter_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    shape = list(input_.shape)
    assert shape[0] % world_size == 0
    shape[0] //= world_size
    output = torch.empty(shape, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.reduce_scatter_tensor(
        output,
        input_.contiguous(),
        group=get_tensor_model_parallel_group()
        )
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the tensor model parallel region."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the tensor model parallel region."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from tensor model parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_, to_model_parallel: bool = True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, to_model_parallel: bool = True):
        ctx.to_model_parallel = to_model_parallel
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.to_model_parallel:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_: torch.Tensor, to_model_parallel: bool = True) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(input_, to_model_parallel)


def reduce_scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


__all__ = [
    "copy_to_tensor_model_parallel_region",
    "reduce_from_tensor_model_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "gather_from_tensor_model_parallel_region",
    "scatter_to_sequence_parallel_region",
    "gather_from_sequence_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
]
