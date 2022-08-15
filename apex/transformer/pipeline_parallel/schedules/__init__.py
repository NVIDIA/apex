from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import (
    forward_backward_no_pipelining,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import (
    _forward_backward_pipelining_with_interleaving,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)

__all__ = [
    "get_forward_backward_func",
]


class ExperimentalWarning(Warning):
    pass


def get_forward_backward_func(
    virtual_pipeline_model_parallel_size, pipeline_model_parallel_size,
):
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if virtual_pipeline_model_parallel_size is not None:
            if get_num_microbatches() % pipeline_model_parallel_size != 0:
                msg = "number of microbatches is not divisible by pipeline-parallel size when using interleaved schedule"
                raise RuntimeError(msg)
            forward_backward_func = _forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func
