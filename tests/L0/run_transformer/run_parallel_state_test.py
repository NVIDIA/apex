import torch

from apex.transformer import parallel_state
from apex.transformer.testing.commons import initialize_distributed


if __name__ == "__main__":
    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if world_size % 2 != 0:
        raise RuntimeError(f"`world_size` is odd: {world_size}")

    data_parallel_size, tensor_parallel_size, pipeline_parallel_size = 1, 1, 1
    virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank = None, None
    if world_size >= 4:
        tensor_parallel_size = 2
        pipeline_parallel_size = world_size // tensor_parallel_size
        virtual_pipeline_model_parallel_size = 2
        pipeline_model_parallel_split_rank = pipeline_parallel_size // 2
    elif world_size > 1:
        tensor_parallel_size = 1
        pipeline_parallel_size = world_size
    else:
        raise RuntimeError(f"World size too small: {world_size}")

    # Check rank and group are set.
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

    for name in func_names:
        value = getattr(parallel_state, name)()
        assert value is not None
