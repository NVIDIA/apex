# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy

from .data import broadcast_data

from .enums import (
    LayerType,
    AttnType,
    AttnMaskType,
)

from .fused_softmax import FusedScaleMaskSoftmax

from .initialize import (
    is_unitialized,
    destroy_model_parallel,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_embedding_group,
    get_model_parallel_group,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_rank,
    set_tensor_model_parallel_rank,
    get_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_tensor_model_parallel_src_rank,
    get_pipeline_model_parallel_first_rank,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_tensor_model_parallel_world_size,
    set_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    set_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_rank,
    set_virtual_pipeline_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from .layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    set_tensor_model_parallel_attributes,
    set_defaults_if_not_set_tensor_model_parallel_attributes,
    copy_tensor_model_parallel_attributes,
)

from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

from .random import (
    checkpoint,
    get_cuda_rng_tracker,
    init_checkpointed_activations_memory_buffer,
    model_parallel_cuda_manual_seed,
    reset_checkpointed_activations_memory_buffer,
    gather_split_1d_tensor,
    split_tensor_into_1d_equal_chunks,
)

from .utils import divide, split_tensor_along_last_dim


__all__ = [
    # cross_entropy.py
    "vocab_parallel_cross_entropy",
    # data.py
    "broadcast_data",
    # enums.py
    "LayerType",
    "AttnType",
    "AttnMaskType",
    # fused_softmax.py
    "FusedScaleMaskSoftmax",
    # initialize.py
    "is_unitialized",
    "destroy_model_parallel",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_embedding_group",
    "get_model_parallel_group",
    "get_tensor_model_parallel_group",
    "get_pipeline_model_parallel_group",
    "get_tensor_model_parallel_rank",
    "set_tensor_model_parallel_rank",
    "get_pipeline_model_parallel_rank",
    "set_pipeline_model_parallel_rank",
    "is_pipeline_first_stage",
    "is_pipeline_last_stage",
    "get_tensor_model_parallel_src_rank",
    "get_pipeline_model_parallel_first_rank",
    "get_pipeline_model_parallel_last_rank",
    "get_pipeline_model_parallel_next_rank",
    "get_pipeline_model_parallel_prev_rank",
    "get_tensor_model_parallel_world_size",
    "set_tensor_model_parallel_world_size",
    "get_pipeline_model_parallel_world_size",
    "set_pipeline_model_parallel_world_size",
    "get_virtual_pipeline_model_parallel_rank",
    "set_virtual_pipeline_model_parallel_rank",
    "initialize_model_parallel",
    "model_parallel_is_initialized",
    # layers.py
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "set_tensor_model_parallel_attributes",
    "set_defaults_if_not_set_tensor_model_parallel_attributes",
    "copy_tensor_model_parallel_attributes",
    # mappings.py
    "copy_to_tensor_model_parallel_region",
    "gather_from_tensor_model_parallel_region",
    "reduce_from_tensor_model_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    # random.py
    "checkpoint",
    "get_cuda_rng_tracker",
    "init_checkpointed_activations_memory_buffer",
    "model_parallel_cuda_manual_seed",
    "reset_checkpointed_activations_memory_buffer",
    "gather_split_1d_tensor",
    "split_tensor_into_1d_equal_chunks",
    # utils.py
    "divide",
    "split_tensor_along_last_dim",
]
