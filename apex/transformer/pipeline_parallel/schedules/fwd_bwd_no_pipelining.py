import torch

from .common import placeholder_handler
from .common import forward_step
from .common import backward_step
from ..utils import get_num_microbatches


def forward_backward_no_pipelining(
    forward_step_func, data_iterator, model, optimizer, timers, forward_only
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.
    """
    assert len(model) == 1
    model = model[0]

    context_handler = placeholder_handler
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        context_handler = model.no_sync

    losses_reduced = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(
                forward_step_func, data_iterator, model, input_tensor, losses_reduced
            )
            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(
        forward_step_func, data_iterator, model, input_tensor, losses_reduced
    )
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    return losses_reduced
