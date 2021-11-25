from typing import Union, List, Optional, Sequence

import torch
import torch.nn.parallel.distributed.DistributedDataParallel as DDP

from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.pipeline_parallel.utils import get_kth_microbatch
from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import unwrap_model
from apex.transformer.pipeline_parallel.schedules.common import Batch, FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import backward_step
from apex.transformer.pipeline_parallel.schedules import utils
from apex.transformer.log_util import get_transformer_logger


__all__ = ["forward_backward_pipelining_without_interleaving"]


_logger = get_transformer_logger(__name__)


def forward_backward_pipelining_without_interleaving(
    forward_step_func: FwdStepFunc,
    batch: Batch,
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    *,
    forward_only: bool,
    tensor_shape: Optional[Union[List[int], torch.Size]] = None,
    decoder_sequence_length: Optional[int] = None,
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

    model: List[torch.nn.Module] = listify_model(model)
    if len(model) != 1:
        msg = f"`model` is expected be a `nn.Module`, but {type(model)}"
        raise RuntimeError(msg)
    model: torch.nn.Module = model[0]

    # Compute number of warmup microbatches.
    num_microbatches: int = get_num_microbatches()
    num_warmup_microbatches: int = (
        parallel_state.get_pipeline_model_parallel_world_size() - parallel_state.get_pipeline_model_parallel_rank() - 1
    )
    num_warmup_microbatches: int = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining: int = num_microbatches - num_warmup_microbatches

    model_type: ModelType = getattr(unwrap_model(model, (DDP,)), "model_type", ModelType.encoder_or_decoder)
    rank: int = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes: List[List[int]] = utils.get_tensor_shapes(rank - 1, model_type, tensor_shape=tensor_shape)
    send_tensor_shapes: List[List[int]] = utils.get_tensor_shapes(rank, model_type, tensor_shape=tensor_shape)

    _logger.info(
        f"num_microbatches: {num_microbatches}, "
        f"num_warmup_microbatches: {num_warmup_microbatches}, "
        f"num_microbatches_remaining: {num_microbatches_remaining}"
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors: List[Union[None, torch.Tensor]] = []
    output_tensors: List[Union[None, torch.Tensor]] = []
    losses_reduced: List[Union[None, torch.Tensor]] = []
    ###################################################################################################################
    # Run warmup forward passes.
    ###################################################################################################################
    _logger.info("Warmup")
    for i in range(num_warmup_microbatches):
        _logger.debug(f"warmup iter: {i} / {num_warmup_microbatches}")
        _logger.debug("receive fwd")
        input_tensor = utils.recv_forward(tensor_shape=recv_tensor_shapes)
        cur_microbatch = get_kth_microbatch(batch, i)
        output_tensor = forward_step(forward_step_func, cur_microbatch, model, input_tensor, losses_reduced)
        _logger.debug("send fwd")
        utils.send_forward(output_tensor, tensor_shape=send_tensor_shapes)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        _logger.debug("recv_forward before steady state start")
        input_tensor: List[Union[None, torch.Tensor]] = utils.recv_forward(tensor_shape=recv_tensor_shapes)

    ###################################################################################################################
    # Run 1F1B in steady state.
    ###################################################################################################################
    _logger.info("Steady phase")
    for i in range(num_microbatches_remaining):
        _logger.debug(f"steady iter: {i} / {num_microbatches_remaining}")
        last_iteration: bool = i == (num_microbatches_remaining - 1)

        cur_microbatch: torch.Tensor = get_kth_microbatch(batch, i + num_warmup_microbatches)
        output_tensor: Union[torch.Tensor, Sequence[torch.Tensor]] = forward_step(
            forward_step_func, cur_microbatch, model, input_tensor, losses_reduced
        )
        if forward_only:
            _logger.debug("send fwd")
            utils.send_forward(output_tensor, tensor_shapes=send_tensor_shapes)

            if not last_iteration:
                _logger.debug("receive fwd (last iteration)")
                input_tensor = utils.recv_forward(tensor_shapes=recv_tensor_shapes)

        else:
            _logger.debug("send fwd & receive bwd")
            output_tensor_grad = utils.send_forward_recv_backward(output_tensor, tensor_shapes=send_tensor_shapes)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

            if last_iteration:
                input_tensor = None
                _logger.debug("send bwd")
                utils.send_backward(input_tensor_grad, tensor_shapes=recv_tensor_shapes)
            else:
                _logger.debug("send bwd and receive fwd")
                input_tensor = p2p_communication.send_backward_recv_forward(
                    input_tensor_grad, tensor_shapes=recv_tensor_shapes
                )
    ###################################################################################################################
    # Run cooldown backward passes.
    ###################################################################################################################
    _logger.info("Cooldown phase")
    if not forward_only:
        for i in range(num_warmup_microbatches):
            _logger.debug(f"cooldown iter: {i} / {num_warmup_microbatches}")
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            _logger.debug("receive bwd")
            output_tensor_grad = utils.recv_backward(tensor_shapes=send_tensor_shapes)

            input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

            _logger.debug("send bwd")
            utils.send_backward(input_tensor_grad, tensor_shapes=recv_tensor_shapes)

    return losses_reduced
