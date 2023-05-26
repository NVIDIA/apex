# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
# TODO (mkozuki): Replace assert with RuntimeError.
# TODO (mkozuki): Sort the functions in the same order of megatron/mpu/initialize.py
"""Model and data parallel groups."""
from typing import Tuple, Optional
import warnings
import os
import torch

from apex.transformer.log_util import get_transformer_logger
from apex.transformer._ucc_util import HAS_UCC


_logger = get_transformer_logger(__name__)

# N.B. (mkozuki): Diff btwn Megatron-LM & apex parallel_state
# set(megatron_mpu_initialize_funcs) - set(apex.transformer.parallel_state) =
# {
#     'get_num_layers',
# }


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Relative position embedding group.
_ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP = None
_DECODER_RELATIVE_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Data parallel AMAX reduction group that the current rank belongs to.
_AMAX_REDUCTION_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the relative position embedding.
_ENCODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS = None
_DECODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage
_PIPELINE_GLOBAL_RANKS = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None

def set_nccl_socket_envs():
    if os.getenv("NCCL_SOCKET_IFNAME") is None:
        raise RuntimeError("NCCL_SOCKET_IFNAME was not set")
    os.environ["NCCL_NET"] = "Socket"

def set_nccl_ib_envs():
    os.environ["NCCL_NET"] = "IB"

def init_nccl_net(group):
    temp = torch.ones(1, device="cuda")
    torch.distributed.all_reduce(temp, group=group)
    torch.cuda.synchronize()

def new_nccl_socket_group(ranks):
    set_nccl_socket_envs()
    group = torch.distributed.new_group(ranks, backend="nccl")
    init_nccl_net(group=group)
    return group

def new_nccl_ib_group(ranks):
    set_nccl_ib_envs()
    group = torch.distributed.new_group(ranks, backend="nccl")
    init_nccl_net(group=group)
    return group

def new_process_group(ranks, backend):
    """
    This function creates process groups.

    In addition to simply creating the process groups, it initializes NCCL
    for hybrid IB/Socket network like in the following diagram:

                            ____________
      [GPU Node 0]---TCP---|            |---TCP---[GPU Node 2]
         |                 |            |            |
         |                 |            |            |
        IB                 | IP Network |           IB
         |                 |            |            |
         |                 |            |            |
      [GPU Node 1]---TCP---|____________|---TCP---[GPU Node 3]


    If an environment variable NUM_GPUS_PER_IB_BLOCK is defined it looks up the ranks
    and determines whether the list of ranks belong to the same computational block where
    GPUs nodes are interconnected via IB type of connection or not.
    If all ranks are in the same block, the process group will use NCCL_NET=IB for
    communication, otherwise it will use NCCL_NET=Socket.

    If NCCL_NET=Socket is ever to be used, the user must set NCCL_SOCKET_IFNAME.
    Additionally, it is recommended to set NCCL_SOCKET_NTHREADS and
    NCCL_NSOCKS_PERTHREAD before running the job.
    See: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
    for more info

    The core assumption for this functionality is that the ranks are evenly divided
    into IB blocks and all these IB blocks are of the same size.
    """
    if backend is None:
        backend = "nccl"

    compute_block_size = os.getenv("NUM_GPUS_PER_IB_BLOCK")
    if backend == "nccl" and compute_block_size is not None:
        compute_block_size = int(compute_block_size)
        blocks = [rank // compute_block_size for rank in ranks]
        use_ib = all(block == blocks[0] for block in blocks)
        if use_ib:
            return new_nccl_ib_group(ranks)
        else:
            return new_nccl_socket_group(ranks)
    else:
        return torch.distributed.new_group(ranks, backend=backend)

def initialize_model_parallel(
    tensor_model_parallel_size_: int = 1,
    pipeline_model_parallel_size_: int = 1,
    virtual_pipeline_model_parallel_size_: Optional[int] = None,
    pipeline_model_parallel_split_rank_: Optional[int] = None,
    use_fp8_: bool = False,
    init_mpi_proc_group: bool = False,
    *,
    default_backend: Optional[str] = None,
    p2p_backend: Optional[str] = None,
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model tensor.
        pipeline_model_parallel_size: number of GPUs used to parallelize model pipeline.
        virtual_pipeline_model_parallel_size: number of virtual stages (interleaved pipeline).
        pipeline_model_parallel_split_rank: for models with both encoder and decoder, rank in pipeline with split point.
        use_fp8_: FP8 training that needs AMAX reduction across data-parallel ranks.
        init_mpi_proc_group: Create a MPI process group, which is used for UCX-based communication APIs.
    Keyword Arguments:
        default_backend: Backend of process groups except for pipeline parallel ones.
            If :obj:`None`, the backend specified in `torch.distributed.init_process_group` will be used.
        p2p_backend: Backend of process groups for pipeline model parallel.
            If :obj:`None`, the backend specified in `torch.distributed.init_process_group` will be used.

    .. note::
        `torch_ucc <https://github.com/facebookresearch/torch_ucc>`_ is
        necessary for "ucc" backend.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    assert default_backend is None or default_backend in ("nccl", "ucc")
    assert p2p_backend is None or p2p_backend in ("nccl", "ucc")
    if "ucc" in (default_backend, p2p_backend):
        if not HAS_UCC:
            raise ImportError("UCC backend requires pytorch source build with UCC installed and enabled")
        warnings.warn("`ucc` backend support is experimental", ExperimentalWarning)
    if default_backend == "ucc":
        warnings.warn("The UCC's functionality as `default_backend` is not well verified", ExperimentalWarning)

    # Saving the NCCL_NET type for reusing it at the epilogue 
    default_nccl_net = os.getenv("NCCL_NET")

    world_size: int = torch.distributed.get_world_size()
    tensor_model_parallel_size: int = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size: int = min(pipeline_model_parallel_size_, world_size)
    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"`world_size` ({world_size}) is not divisible by tensor_model_parallel_size ({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )
    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size
    )
    if torch.distributed.get_rank() == 0:
        _logger.info(
            "> initializing tensor model parallel with size {}".format(
                tensor_model_parallel_size
            )
        )
        _logger.info(
            "> initializing pipeline model parallel with size {}".format(
                pipeline_model_parallel_size
            )
        )
        _logger.info(
            "> initializing data parallel with size {}".format(data_parallel_size)
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size_ is not None:
        # n.b. (eqy) This check was inherited from Megatron-LM, need to revisit
        # the root cause as we do see numerical mismatches with 2 stages and
        # the interleaved schedule
        assert pipeline_model_parallel_size_ > 2, (
            "pipeline-model-parallel size should be greater than 2 with "
            "interleaved schedule"
        )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
            virtual_pipeline_model_parallel_size_
        )

    if pipeline_model_parallel_split_rank_ is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank_

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = new_process_group(ranks, backend=default_backend)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group

    # Build the amax-reduction groups for fp8 precision conversion.
    if use_fp8_:
        global _AMAX_REDUCTION_GROUP
        assert _AMAX_REDUCTION_GROUP is None, "amax reduction group is already initialized"
        amax_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_amax_groups: int = world_size // amax_group_size
        for i in range(num_amax_groups):
            start_rank = i * amax_group_size
            end_rank = (i + 1) * amax_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks, backend=default_backend)
            if rank in ranks:
                _AMAX_REDUCTION_GROUP = group

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        ranks = [
            data_parallel_group_ranks[i]
            for data_parallel_group_ranks in all_data_parallel_group_ranks
        ]
        group = new_process_group(ranks, backend=default_backend)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        )
        group = new_process_group(ranks, backend=default_backend)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), "pipeline model parallel group is already initialized"
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert (
        _POSITION_EMBEDDING_GROUP is None
    ), "position embedding group is already initialized"
    global _ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP
    global _DECODER_RELATIVE_POSITION_EMBEDDING_GROUP
    global _ENCODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS
    global _DECODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS
    assert _ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP is None or \
           _DECODER_RELATIVE_POSITION_EMBEDDING_GROUP is None, \
        'relative position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = new_process_group(ranks, backend=p2p_backend)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        encoder_relative_position_embedding_ranks = None
        decoder_relative_position_embedding_ranks = None
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            encoder_relative_position_embedding_ranks = [ranks[0]]
            decoder_relative_position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank_ is not None:
                encoder_relative_position_embedding_ranks = \
                    ranks[:pipeline_model_parallel_split_rank_]
                decoder_relative_position_embedding_ranks = \
                    ranks[pipeline_model_parallel_split_rank_:]
                if ranks[pipeline_model_parallel_split_rank_] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank_],
                        ranks[-1],
                    ]
                if (
                    ranks[pipeline_model_parallel_split_rank_]
                    not in position_embedding_ranks
                ):
                    position_embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank_],
                    ]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks
            encoder_relative_position_embedding_ranks = ranks
            decoder_relative_position_embedding_ranks = ranks

        group = new_process_group(embedding_ranks, backend=p2p_backend)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = new_process_group(position_embedding_ranks, backend=p2p_backend)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        if encoder_relative_position_embedding_ranks:
            group = new_process_group(encoder_relative_position_embedding_ranks, backend=p2p_backend)
        if rank in encoder_relative_position_embedding_ranks:
            _ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _ENCODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS = \
                encoder_relative_position_embedding_ranks

        if decoder_relative_position_embedding_ranks:
            group = new_process_group(decoder_relative_position_embedding_ranks, backend=p2p_backend)
        if rank in decoder_relative_position_embedding_ranks:
            _DECODER_RELATIVE_POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _DECODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS = \
                decoder_relative_position_embedding_ranks

    if init_mpi_proc_group:
        torch.distributed.new_group(backend='mpi')

    if default_nccl_net == "Socket":
        set_nccl_socket_envs()
    elif default_nccl_net == "IB":
        set_nccl_ib_envs()
    elif default_nccl_net is None:
        os.unsetenv("NCCL_NET")
    else:
        os.environ["NCCL_NET"] = default_nccl_net

def get_rank_info() -> Tuple[int, int, int]:
    """Returns a tuple of (data, tensor, pipeline, virtual pipeline)-parallel-rank for logger."""
    if model_parallel_is_initialized():
        return (
            get_data_parallel_rank(),
            get_tensor_model_parallel_rank(),
            get_pipeline_model_parallel_rank(),
            get_virtual_pipeline_model_parallel_rank(),
        )
    return (0, 0, 0, 0)


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
    ), "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_amax_reduction_group():
    """Get the amax reduction group the caller rank belongs to."""
    assert _AMAX_REDUCTION_GROUP is not None, \
        "AMAX reduction group is not initialized"
    return _AMAX_REDUCTION_GROUP


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert (
        _POSITION_EMBEDDING_GROUP is not None
    ), "position embedding group is not initialized"
    return _POSITION_EMBEDDING_GROUP

def get_encoder_relative_position_embedding_group():
    """Get the encoder relative position embedding group the caller rank belongs to."""
    assert _ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP is not None, \
        'encoder relative position embedding group is not initialized'
    return _ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP

def get_decoder_relative_position_embedding_group():
    """Get the decoder relative position embedding group the caller rank belongs to."""
    assert _DECODER_RELATIVE_POSITION_EMBEDDING_GROUP is not None, \
        'decoder relative position embedding group is not initialized'
    return _DECODER_RELATIVE_POSITION_EMBEDDING_GROUP

def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return whether the current rank is in position embedding group."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS

def is_rank_in_encoder_relative_position_embedding_group():
    """Return true if current rank is in encoder relative position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _ENCODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _ENCODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS

def is_rank_in_decoder_relative_position_embedding_group():
    """Return true if current rank is in decoder relative position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _DECODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _DECODER_RELATIVE_POSITION_EMBEDDING_GLOBAL_RANKS

def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(
        rank + 1
    )


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


# TODO (mkozuki): Add [`get_num_layers`](https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/mpu/initialize.py#L321) here, maybe?


def get_pipeline_model_parallel_split_rank():
    """Return my rank for the pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def set_pipeline_model_parallel_split_rank(pipeline_model_parallel_split_rank: int):
    """Set my rank for the pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1
    )


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def set_virtual_pipeline_model_parallel_world_size(size):
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = size


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    data_parallel_size: int = get_data_parallel_world_size()
    num_data_parallel_groups = torch.distributed.get_world_size() // data_parallel_size
    return global_rank % num_data_parallel_groups


def get_pipeline_model_parallel_first_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


# note (mkozuki): `destroy_model_parallel` voids more global variables than Megatron-LM.
# Otherwise pipeline parallel forward_backward functions test hangs possibly because
# the clean-up of the original is NOT enough.
def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _AMAX_REDUCTION_GROUP
    _AMAX_REDUCTION_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP
    _ENCODER_RELATIVE_POSITION_EMBEDDING_GROUP = None
    global _DECODER_RELATIVE_POSITION_EMBEDDING_GROUP
    _DECODER_RELATIVE_POSITION_EMBEDDING_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None


# Used to warn when the UCC is specified.
class ExperimentalWarning(Warning): pass
