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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
from typing import Optional, Dict, Tuple, List
import warnings

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.parallel_state import get_tensor_model_parallel_group
from apex.transformer.parallel_state import get_tensor_model_parallel_rank
from apex.transformer.parallel_state import get_tensor_model_parallel_world_size
from apex.transformer.utils import divide
from apex.transformer.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
)
from apex.transformer.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from apex.transformer.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from apex.transformer.tensor_parallel.mappings import (
    scatter_to_tensor_model_parallel_region,
)
from apex.transformer.tensor_parallel.mappings import (
    reduce_scatter_to_sequence_parallel_region,
)
from apex.transformer.tensor_parallel.random import get_cuda_rng_tracker
from apex.transformer.tensor_parallel.utils import VocabUtility
from apex.transformer.log_util import get_transformer_logger


_logger = get_transformer_logger(__name__)


_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def param_is_not_tensor_parallel_duplicate(param: torch.Tensor) -> bool:
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int) -> None:
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor: torch.Tensor) -> None:
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor: torch.Tensor, source_tensor: torch.Tensor) -> None:
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU.

    Args:
        weight (Parameter):
        init_method (Callable[[Tensor], None]): Taking a Tensor and initialize its elements.
        partition_dim (int): Dimension to apply partition.
        stride (int):
    """

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    with get_cuda_rng_tracker().fork():
        init_method(weight)


# TODO (mkozuki): Re-consider removing params_dtype from arguments to make this
# more parallel with _initialize_affine_weight_gpu
def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    # Initialize master weight
    master_weight = torch.empty(
        output_size, input_size, dtype=torch.float, requires_grad=False
    )
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_method=init.xavier_normal_,
        *,
        params_dtype: torch.dtype=torch.float32,
        use_cpu_initialization: bool = False,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings,
            get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
                params_dtype=params_dtype,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=1
            )

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication and gradient accumulation fusion in backprop."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        gradient_accumulation_fusion: bool,
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        use_16bit_in_wgrad_accum_fusion: bool = False,
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel_enabled = sequence_parallel_enabled
        ctx.use_16bit_in_wgrad_accum_fusion = use_16bit_in_wgrad_accum_fusion

        if ctx.sequence_parallel_enabled:
            world_size = get_tensor_model_parallel_world_size()
            # `input` is supposed to be 3D and its order of dimension is [sequence, batch, hidden]
            shape = list(input.shape)
            shape[0] *= world_size

            all_gather_buffer = torch.empty(
                shape,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            torch.distributed._all_gather_base(all_gather_buffer, input, group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.sequence_parallel_enabled:
            world_size = get_tensor_model_parallel_world_size()
            shape = list(input.shape)
            shape[0] *= world_size

            all_gather_buffer = torch.empty(
                shape,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(),
                async_op=True,
            )
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel_enabled:
            handle.wait()

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )

        if ctx.sequence_parallel_enabled:
            assert not ctx.async_grad_allreduce
            sub_grad_input = torch.empty(input.shape, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False)
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input,
                grad_input,
                group=get_tensor_model_parallel_group(),
                async_op=True
            )

        if ctx.gradient_accumulation_fusion:
            if not ctx.use_16bit_in_wgrad_accum_fusion:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_output, weight.main_grad
                )
            else:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_input, grad_output, weight.main_grad
                )
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)

        grad_bias = grad_output.sum(dim=0) if use_bias else None
        if ctx.sequence_parallel_enabled:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None
        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
) -> torch.Tensor:
    args = _cast_if_autocast_enabled(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
        False,  # use_16bit_in_wgrad_accum_fusion
    )
    with torch.cuda.amp.autocast(enabled=False):
        return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


def linear_with_grad_accumulation_and_async_allreduce_in16bit(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
) -> torch.Tensor:
    args = _cast_if_autocast_enabled(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
        True,  # use_16bit_in_wgrad_accum_fusion
    )
    with torch.cuda.amp.autocast(enabled=False):
        return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be sequence, batch, and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.

    Keyword Arguments:
        no_async_tensor_model_parallel_allreduce:
        params_dtype:
        use_cpu_initialization:
        gradient_accumulation_fusion:
        accumulation_in_fp16:
        sequence_parallel_enabled:
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        *,
        no_async_tensor_model_parallel_allreduce=False,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        accumulation_in_fp16: bool = False,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(self.output_size_per_partition, self.input_size, dtype=params_dtype)
            )
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
                params_dtype=params_dtype,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=stride)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.async_tensor_model_parallel_allreduce = (
            not no_async_tensor_model_parallel_allreduce and world_size > 1
        )
        if sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(
                    f"`sequence_parallel_enabled` is set to `True`, but got world_size of {world_size}"
                )
                # sequence_parallel_enabled = False
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if gradient_accumulation_fusion:
            if not _grad_accum_fusion_available:
                # Basically, apex.transformer module users are expected to install APEX's
                # `--cpp_ext` and `--cuda_ext`. The example installation command is as follows:
                # `pip install --global-option="--cpp_ext" --global-option="--cuda_ext ."
                # at the root of APEX repository.
                warnings.warn(
                    "`gradient_accumulation_fusion` is set to `True` but "
                    "the custom CUDA extension of `fused_weight_gradient_mlp_cuda` module not "
                    "found. Thus `gradient_accumulation_fusion` set to `False`. "
                    "Note that the extension requires CUDA>=11."
                )
                gradient_accumulation_fusion = False
        self.gradient_accumulation_fusion = gradient_accumulation_fusion


        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError("`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot be enabled at the same time.")

        self._forward_impl = (
            linear_with_grad_accumulation_and_async_allreduce_in16bit
            if accumulation_in_fp16
            else linear_with_grad_accumulation_and_async_allreduce
        )

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be sequence, batch, and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    Keyword Arguments:
        params_dtype:
        use_cpu_initialization:
        gradient_accumulation_fusion:
        accumulation_in_fp16:
        sequence_parallel_enabled:
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        *,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        accumulation_in_fp16: bool = False,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")

        # as an argument to this function?
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=params_dtype
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
                params_dtype=params_dtype,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=stride
            )
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=params_dtype,
                    )
                )
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
            setattr(self.bias, "sequence_parallel_enabled", sequence_parallel_enabled)
        else:
            self.register_parameter("bias", None)

        self._forward_impl = (
            linear_with_grad_accumulation_and_async_allreduce_in16bit
            if accumulation_in_fp16
            else linear_with_grad_accumulation_and_async_allreduce
        )

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )
        # All-reduce across all the partitions.
        if self.sequence_parallel_enabled:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
