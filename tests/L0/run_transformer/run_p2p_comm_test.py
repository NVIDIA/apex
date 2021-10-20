import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.p2p_communication import recv_forward
from apex.transformer.pipeline_parallel.p2p_communication import send_backward
from apex.transformer.pipeline_parallel.p2p_communication import send_forward_recv_backward
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator
from apex.transformer.utils import rank_print


global_vars.set_global_variables()


def run_test(pipeline_model_parallel_size):
    parallel_state.initialize_model_parallel(1, pipeline_model_parallel_size, None)

    shape = [3]
    rank = parallel_state.get_pipeline_model_parallel_rank()
    tensor = torch.full(shape, rank, dtype=torch.float).cuda()
    next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    expected = torch.ones_like(tensor) * next_rank

    # rank_print("Start `recv_forward`, first rank does nothing here")
    inp = recv_forward(shape)
    # rank_print("Finish `recv_forward`")

    # rank_print("Start `send_forward_recv_backward`")
    grad = send_forward_recv_backward(tensor, shape, dtype=torch.float)
    # rank_print("Finish `send_forward_recv_backward`")

    # rank_print("Start `send_backward`, first rank does nothing here")
    send_backward(tensor, shape, dtype=torch.float)
    # rank_print("Finish `recv_forward`")

    if not parallel_state.is_pipeline_last_stage():
        assert torch.equal(expected, grad)
    else:
        # rank_print("Last rank so `send_forward_recv_backward` does nothing")
        assert grad is None
    # rank_print("Exiting `run_test` function")


if __name__ == "__main__":
    initialize_distributed()

    pipeline_model_parallel_size = torch.distributed.get_world_size()
    run_test(pipeline_model_parallel_size)
