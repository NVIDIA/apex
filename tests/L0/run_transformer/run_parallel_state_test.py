from typing import Sequence, Optional

import torch

from apex.transformer import parallel_state
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator


def run(
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    virtual_pipeline_model_parallel_size: Optional[int],
    pipeline_model_parallel_split_rank: Optional[int],
) -> bool:
    # Initialize model parallel
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size_=tensor_parallel_size,
        pipeline_model_parallel_size_=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size_=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank_=pipeline_model_parallel_split_rank,
    )

    # This test always sets Data Parallel World Size to 1.
    pipeline_model_parallel_rank = torch.distributed.get_rank()
    _pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    _tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
    # TODO (mkozuki): Add some checks on pipeline rank, ideally.
    pipeline_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
    pipeline_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()

    embedding_ranks = [pipeline_first_rank, pipeline_last_rank]
    position_embedding_ranks = [pipeline_first_rank]
    if pipeline_model_parallel_split_rank is not None:
        embedding_ranks.append(pipeline_model_parallel_split_rank)
        position_embedding_ranks.append(pipeline_model_parallel_split_rank)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            f"### Expected embedding ranks: {embedding_ranks} & "
            f"position_embedding_ranks: {position_embedding_ranks}"
        )

    should_have_embedding = pipeline_model_parallel_rank in embedding_ranks
    should_have_position_embedding = pipeline_model_parallel_rank in position_embedding_ranks

    failures = []
    for name in parallel_state._getter_functions:
        if virtual_pipeline_model_parallel_size is None:
            if name in (
                "get_virtual_pipeline_model_parallel_rank",
                "get_virtual_pipeline_model_parallel_world_size",
            ):
                continue
        if pipeline_model_parallel_split_rank is None:
            if name == "get_pipeline_model_parallel_split_rank":
                continue
        try:
            value = getattr(parallel_state, name)()
        except Exception:
            is_failure = True
            # Allow some tests to fail on certain ranks.
            # - `get_embedding_group`
            # - `get_position_embedding_group`
            # on ranks that are **NOT** pipeline_model_parallel_(first|last)_rank
            # nor pipeline_model_parallel_split_rank (if applicable).
            if (
                name == "get_embedding_group" and
                pipeline_model_parallel_rank not in embedding_ranks
            ):
                is_failure = False
            if (
                name == "get_position_embedding_group" and
                pipeline_model_parallel_rank not in position_embedding_ranks
            ):
                is_failure = False
            if is_failure:
                failures.append(name)
        else:
            if value is None:
                failures.append(name)
    torch.distributed.barrier()
    if len(failures) > 0:
        msg = f"[Rank - {parallel_state.get_rank_info()}] {len(failures)} / {len(parallel_state._getter_functions)} failed: {failures}"
        print(msg, flush=True)
    parallel_state.destroy_model_parallel()

    return failures


if __name__ == "__main__":
    global_vars.set_global_variables()
    initialize_distributed()
    args = global_vars.get_args()
    world_size = torch.distributed.get_world_size()
    if world_size % 2 != 0:
        raise RuntimeError(f"`world_size` shouldn't be odd: {world_size}")

    data_parallel_size, tensor_parallel_size, pipeline_model_parallel_size = 1, 1, 1
    virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank = None, None
    if world_size >= 4:
        # TODO (mkozuki): Support pipeline/tensor/data parallel case.
        # if world_size > 4:
        #     tensor_parallel_size = 2
        pipeline_model_parallel_size = world_size // tensor_parallel_size
        virtual_pipeline_model_parallel_size = 2
        pipeline_model_parallel_split_rank = pipeline_model_parallel_size // 2
    else:
        raise RuntimeError(f"World size too small: {world_size}")

    virtual_pipeline_model_parallel_size_values = [None]
    if virtual_pipeline_model_parallel_size is not None:
        virtual_pipeline_model_parallel_size_values.append(virtual_pipeline_model_parallel_size)
    pipeline_model_parallel_split_rank_values = [None]
    if pipeline_model_parallel_split_rank is not None:
        pipeline_model_parallel_split_rank_values.append(pipeline_model_parallel_split_rank)

    failures = {}
    for i, (virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank) in enumerate(
        zip(virtual_pipeline_model_parallel_size_values, pipeline_model_parallel_split_rank_values)
    ):
        print_separator(f"Case {i}")
        print_separator(
            f"`virtual_pipeline_model_parallel_size`: {virtual_pipeline_model_parallel_size} "
            f"`pipeline_model_parallel_split_rank`: {pipeline_model_parallel_split_rank}"
        )
        ret = run(
            tensor_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
        )
        failures[i] = ret

    torch.distributed.barrier()
    success = True
    for key, values in failures.items():
        if len(values) > 0:
            success = False
    if not success:
        raise RuntimeError(f"Rank {torch.distributed.get_rank()}: Test Failed! -- {failures}")
    else:
        print('>> passed the test :-)')
