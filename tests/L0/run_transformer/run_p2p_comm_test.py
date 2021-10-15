import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.p2p_communication import send_forward_recv_backward
from apex.transformer.pipeline_parallel.schedules.common import rank_print
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator


global_vars.set_global_variables()


def run_test(pipeline_model_parallel_size):
    parallel_state.initialize_model_parallel(1, pipeline_model_parallel_size, None)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    shape = [3]
    tensor = torch.full(shape, rank).cuda()

    rank_print("Start `send_forward_recv_backward`")
    grad = send_forward_recv_backward(tensor, shape, dtype=torch.float)
    rank_print("Finish `send_forward_recv_backward`")
    next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    expected = torch.ones_like(tensor) * next_rank

    assert torch.equal(expected, grad)
    rank_print("Exitting `run_test` function")


if __name__ == "__main__":
    initialize_distributed()

    pipeline_model_parallel_size = torch.distributed.get_world_size()
    run_test(pipeline_model_parallel_size)
