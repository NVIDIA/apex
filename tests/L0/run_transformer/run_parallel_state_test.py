from typing import Sequence

import torch

from apex.transformer import parallel_state
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator


def run(pipeline_model_parallel_split_rank) -> bool:
    # Initialize model parallel
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size_=tensor_parallel_size,
        pipeline_model_parallel_size_=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size_=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank_=pipeline_model_parallel_split_rank,
    )

    # Check if rank and group are set.
    func_names = (
        # Group
        "get_model_parallel_group",
        "get_tensor_model_parallel_group",
        "get_pipeline_model_parallel_group",
        "get_data_parallel_group",
        "get_embedding_group",
        "get_position_embedding_group",
        # World size
        "get_virtual_pipeline_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
        "get_tensor_model_parallel_world_size",
        "get_data_parallel_world_size",
        # Rank
        "get_virtual_pipeline_model_parallel_rank",
        "get_pipeline_model_parallel_first_rank",
        "get_pipeline_model_parallel_prev_rank",
        "get_pipeline_model_parallel_next_rank",
        "get_pipeline_model_parallel_last_rank",
        "get_tensor_model_parallel_src_rank",
        "get_pipeline_model_parallel_split_rank",
        "get_data_parallel_src_rank",
        "get_pipeline_model_parallel_rank",
        "get_tensor_model_parallel_rank",
        "get_data_parallel_rank",
    )

    # This test always sets Data Parallel Worldsize to 1.
    pipeline_model_parallel_rank = torch.distributed.get_rank()
    assert pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank()
    pipeline_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
    pipeline_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()

    embedding_ranks = [pipeline_first_rank, pipeline_last_rank]
    position_embedding_ranks = [pipeline_first_rank]
    if pipeline_model_parallel_split_rank is not None:
        embedding_ranks.append(pipeline_model_parallel_split_rank)
        position_embedding_ranks.append(pipeline_model_parallel_split_rank)

    if torch.distributed.get_rank() == 0:
        print(
            f"### Expected embedding ranks: {embedding_ranks} & "
            f"position_embedding_ranks: {position_embedding_ranks}"
        )

    should_have_embedding = pipeline_model_parallel_rank in embedding_ranks
    should_have_position_embedding = pipeline_model_parallel_rank in position_embedding_ranks

    failures = []
    for name in func_names:
        if pipeline_model_parallel_split_rank is None:
            if name == "get_pipeline_model_parallel_split_rank":
                print(f"Skip `{name}` as `pipeline_model_parallel_split_rank` is `None`")
                continue
        try:
            value = getattr(parallel_state, name)()
        except Exception:
            is_failure = True
            # Allow some tests to fail on certain ranks.
            # - `get_embedding_group`
            # - `get_position_embedding_group`
            # on ranks that are **NOT** pipeline_model_parallel_(first|last)_rank nor
            # nor pipeline_model_parallel_split_rank (if applicable).
            if (
                name == "get_embedding_group" and
                pipeline_model_parallel_rank not in embedding_ranks
            ):
                print(f"{parallel_state.get_rank_info()} is allowed to fail {name}")
                is_failure = False
            if (
                name == "get_position_embedding_group" and
                pipeline_model_parallel_rank not in position_embedding_ranks
            ):
                print(f"{parallel_state.get_rank_info()} is allowed to fail {name}")
                is_failure = False
            if is_failure:
                print(f"{parallel_state.get_rank_info()} {name} threw")
                failures.append(name)
        else:
            if value is None:
                print(f"{parallel_state.get_rank_info()} {name} failed")
                failures.append(name)
    if failures:
        msg = f"[Rank - {parallel_state.get_rank_info()}] {len(failures)} / {len(func_names)} failed: {failures}"
        print(msg)
    parallel_state.destroy_model_parallel()

    return not bool(failures)


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
        if world_size > 4:
            tensor_parallel_size = 2
        pipeline_model_parallel_size = world_size // tensor_parallel_size
        virtual_pipeline_model_parallel_size = 2
        pipeline_model_parallel_split_rank = pipeline_model_parallel_size // 2
    else:
        raise RuntimeError(f"World size too small: {world_size}")

    pipeline_model_parallel_split_rank_values = [None]
    if pipeline_model_parallel_split_rank is not None:
        pipeline_model_parallel_split_rank_values.append(pipeline_model_parallel_split_rank)

    results = []
    for pipeline_model_parallel_split_rank in pipeline_model_parallel_split_rank_values:
        print_separator(
            f"`pipeline_model_parallel_split_rank`: {pipeline_model_parallel_split_rank}")
        success = run(pipeline_model_parallel_split_rank)
        results.append(success)
    if not all(results):
        raise RuntimeError("Test Failed!")
    else:
        print("### PASS!")
