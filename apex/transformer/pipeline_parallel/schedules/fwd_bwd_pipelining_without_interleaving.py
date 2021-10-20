from typing import Union, List, Optional

import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.pipeline_parallel.utils import get_kth_microbatch
from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.schedules.common import Batch, FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import backward_step
from apex.transformer.utils import rank_print


def forward_backward_pipelining_without_interleaving(
        forward_step_func: FwdStepFunc,
        batch: Batch,
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        *,
        forward_only: bool,
        tensor_shape: Optional[Union[List[int], torch.Size]] = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline stages.

    This pipeline parallel scheduling consists of three steps:
        1. warmup
        2. 1F1B a.k.a. steady state
        3. cooldown if not forward_only

    Args:
        forward_step_func: A function which takes a minibatch and model as its arguments and
            returns model's forward output and the loss function.
            The loss function is supposed to take one `torch.Tensor` and
            return a `torch.Tensor` of loss and a dictionary of `str` and `torch.Tensor`.
        batch: A minibatch, i.e., a list of `torch.Tensor`'s.
        model: A `torch.nn.Module` or a list of `torch.nn.Module`.

    Keyword args:
        forward_only:
        tensor_shape: Shape of tensor. Required for P2P communication.

    Returns:
        a list of loss `torch.Tensor`s if the last stage, empty list otherwise.
    """
    # timers = get_timers()

    model = listify_model(model)
    if len(model) != 1:
        msg = f"`model` is expected be a `nn.Module`, but {type(model)}"
        raise RuntimeError(msg)
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

    # TODO (mkozuki): Remove once debug gets done
    print(
        f">>> rank: {torch.distributed.get_rank()}, "
        f"num_microbatches: {num_microbatches}, "
        f"num_warmup_microbatches: {num_warmup_microbatches}, "
        f"num_microbatches_remaining: {num_microbatches_remaining} -- "
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    losses_reduced = []

    ###################################################################################################################
    # Run warmup forward passes.
    ###################################################################################################################
    # rank_print(f"warmup: {num_warmup_microbatches}")
    for i in range(num_warmup_microbatches):
        input_tensor = p2p_communication.recv_forward(tensor_shape=tensor_shape)
        cur_microbatch = get_kth_microbatch(batch, i)
        output_tensor = forward_step(forward_step_func, cur_microbatch, model, input_tensor, losses_reduced)
        p2p_communication.send_forward(output_tensor, tensor_shape=tensor_shape)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
        # rank_print(f"warmup iter: {i + 1} / {num_warmup_microbatches}")
    # rank_print("warmup done")

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    # rank_print(f"num microbatches remaining: {num_microbatches_remaining}")
    if num_microbatches_remaining > 0:
        # rank_print(f"recv_forward before steady state start")
        input_tensor = p2p_communication.recv_forward(tensor_shape=tensor_shape)
        # rank_print(f"recv_forward before steady state done")

    ###################################################################################################################
    # Run 1F1B in steady state.
    ###################################################################################################################
    # rank_print(f"steady: {num_microbatches_remaining} iters")
    for i in range(num_microbatches_remaining):
        # rank_print(f"steady: iter {i + 1} / {num_microbatches_remaining} iters")
        # if not forward_only:
        #     rank_print(f"len(input_tensors) = {len(input_tensors)}, len(output_tensors) = {len(output_tensors)}")
        last_iteration = i == (num_microbatches_remaining - 1)

        cur_microbatch = get_kth_microbatch(batch, i + num_warmup_microbatches)
        output_tensor = forward_step(forward_step_func, cur_microbatch, model, input_tensor, losses_reduced)
        if forward_only:
            # rank_print(f"steady, no backward: `send_forward` start")
            p2p_communication.send_forward(output_tensor, tensor_shape=tensor_shape)

            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(tensor_shape=tensor_shape)
            # rank_print(f"steady, no backward: `send_forward` finish")

        else:
            # rank_print("L.124 steady, backward: `send_forward_recv_backward` start")
            output_tensor_grad = p2p_communication.send_forward_recv_backward(output_tensor, tensor_shape=tensor_shape)
            # rank_print("L.124 steady, backward: `send_forward_recv_backward` finish")

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )

            if last_iteration:
                input_tensor = None
                # rank_print(f"L.142 steady backward last iteration: `send_backward` start")
                p2p_communication.send_backward(input_tensor_grad, tensor_shape=tensor_shape)
                # rank_print(f"L.142 steady backward last iteration: `send_backward` finish")
            else:
                # rank_print(f"L.146 steady backward: `send_backward_recv_forward` start")
                input_tensor = p2p_communication.send_backward_recv_forward(
                    input_tensor_grad, tensor_shape=tensor_shape)
                # rank_print(f"L.146 steady backward: `send_backward_recv_forward` finish")
    # rank_print(f"steady: exit")
    ###################################################################################################################
    # Run cooldown backward passes.
    ###################################################################################################################
    if not forward_only:
        # rank_print(f"cooldownk: {num_warmup_microbatches} iters")
        for i in range(num_warmup_microbatches):
            # rank_print(f"cooldown iter: {i + 1} / {num_warmup_microbatches}")
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # rank_print(f"cooldown waiting for grad tensor")
            output_tensor_grad = p2p_communication.recv_backward(tensor_shape=tensor_shape)

            # rank_print(f"cooldown received grad tensor")
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )

            # rank_print(f"cooldown sending grad tensor")
            p2p_communication.send_backward(input_tensor_grad, tensor_shape=tensor_shape)
        # rank_print(f"cooldownk exit")

    return losses_reduced
