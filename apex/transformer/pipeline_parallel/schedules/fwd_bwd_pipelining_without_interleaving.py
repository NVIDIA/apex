from typing import List, Union

import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.schedules.common import Batch, FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import backward_step


def forward_backward_pipelining_without_interleaving(
        forward_step_func: FwdStepFunc,
        batch: Batch,
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        forward_only: bool,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns list of losses if the last stage, empty dict otherwise.
    """
    # timers = get_timers()

    model = listify_model(model)
    if len(model) != 1:
        raise RuntimeError("`model` for pipeline model parallel without interleaving must be a `nn.Module`")
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    losses_reduced = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = p2p_communication.recv_forward()
        output_tensor = forward_step(forward_step_func, batch, model, input_tensor, losses_reduced)
        p2p_communication.send_forward(output_tensor)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward()

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        output_tensor = forward_step(forward_step_func, batch, model, input_tensor, losses_reduced)
        if forward_only:
            p2p_communication.send_forward(output_tensor)

            if not last_iteration:
                input_tensor = p2p_communication.recv_forward()

        else:
            output_tensor_grad = p2p_communication.send_forward_recv_backward(output_tensor)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad)
            else:
                input_tensor = p2p_communication.send_backward_recv_forward(input_tensor_grad)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communication.recv_backward()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )

            p2p_communication.send_backward(input_tensor_grad)

    return losses_reduced
