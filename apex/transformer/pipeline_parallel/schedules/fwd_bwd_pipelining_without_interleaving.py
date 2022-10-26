import contextlib
from typing import Any, List, Optional, Sequence, Union
import warnings

import torch

from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.pipeline_parallel.p2p_communication import FutureTensor
from apex.transformer.pipeline_parallel.utils import get_kth_microbatch
from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import get_model_type
from apex.transformer.pipeline_parallel.schedules.common import Batch
from apex.transformer.pipeline_parallel.schedules.common import FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import backward_step
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import free_output_tensor
from apex.transformer.log_util import get_transformer_logger


__all__ = ["forward_backward_pipelining_without_interleaving"]


_logger = get_transformer_logger(__name__)


def get_tensor_shapes(
    rank: int,
    model_type: ModelType,
    *,
    tensor_shape: Union[List[int], torch.Size],
    decoder_sequence_length: Optional[int] = None,
    sequence_parallel_enabled: bool = False,
) -> Sequence[Sequence[int]]:
    """Get tensors shapes

    Args:
        rank: pipeline parallel rank
        model_type:

    Keyword Args:
        tensor_shape:
        decoder_sequence_length:
        sequence_parallel_enabled:
    """
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"

    sequence_length, micro_batch_size, hidden_size = tensor_shape

    tensor_shapes = []

    if sequence_parallel_enabled:
        seq_length = sequence_length // parallel_state.get_tensor_model_parallel_world_size()
    else:
        seq_length = sequence_length

    if model_type == ModelType.encoder_and_decoder:

        if sequence_parallel_enabled:
            dec_seq_length = decoder_sequence_length // parallel_state.get_tensor_model_parallel_world_size()
        else:
            dec_seq_length = decoder_sequence_length

        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
        else:
            tensor_shapes.append((dec_seq_length, micro_batch_size, hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, hidden_size))

    return tensor_shapes


def recv_forward(
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
) -> List[Union[None, torch.Tensor, FutureTensor]]:
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(
                p2p_communication.recv_forward(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                )
            )
    return input_tensors


def recv_backward(
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
) -> List[Union[None, torch.Tensor, FutureTensor]]:
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(
                p2p_communication.recv_backward(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                )
            )
    return output_tensor_grads


def send_forward(
    output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
) -> None:
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(
            output_tensor,
            tensor_shape=tensor_shape,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
        )


def send_backward(
    input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
) -> None:
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(
            input_tensor_grad,
            tensor_shape=tensor_shape,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
        )


def send_forward_recv_backward(
    output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
) -> List[Union[None, torch.Tensor, FutureTensor]]:
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor,
            tensor_shape=tensor_shape,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(
    input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
) -> List[Union[None, torch.Tensor, FutureTensor]]:
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad,
            tensor_shape=tensor_shape,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
        )
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(
    forward_step_func: FwdStepFunc,
    batch: Optional[Batch],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    *,
    forward_only: bool,
    tensor_shape: Optional[Union[List[int], torch.Size]] = None,
    decoder_sequence_length: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    disable_autocast: bool = False,
    deallocate_pipeline_outputs: bool = False,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    custom_sync_context_handler: Optional[Any] = None,
    custom_grad_sync_func: Optional[Any] = None,
    sync_batch_comm: bool = True,
    num_micro_batches_with_partial_activation_checkpoints: Optional[int] = None,
    **kwargs,
) -> List[Union[torch.Tensor, Sequence[torch.Tensor]]]:
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
        tensor_shape: Shape of tensor. The tensor is expected to be 3D and its order of dimension
            is supposed to be ``(sequence, batch, hidden)``.
        dtype: dtype used in p2p communication. If ``None`` (default value),
            torch.float32 will be used even if ``autocast`` is enabled.
        grad_scaler:
        disable_autocast:
        deallocate_pipeline_outputs: If :obj:`True`, free the data of the output tensor of
            each pipeline stage. Experimental.
        sequence_parallel_enabled: Set to :obj:`True` for this function to handle sequence length.
            When :obj:`True`, the sequence length on each tensor model parallel rank is updated
            to :math:`original\_sequence\_length / tensor\_model\_parallel\_world\_size`.
        custom_sync_context_handler: Does nothing if ``None`` (default
            value). Otherwise, a function to construct a context
            manager that disable asynchronous gradient reductions.
            Asynchronous gradient reductions are only enabled in the
            first pipeline stage, during the last backward pass.
        custom_grad_sync_func: Does nothing if ``None`` (default
            value). Otherwise, a function to perform gradient
            reductions. This is called in all pipeline stages except
            the first, during the bubble overhead.
        sync_batch_comm: If :obj:`False`, disable cuda synchronization after the batched communication.
            To disable, https://github.com/pytorch/pytorch/pull/82450 would be required.
        num_micro_batches_with_partial_activation_checkpoints: If :obj:`int`, set the number of
            micro-batches checkpointing the activation of partial number of Transformer layers.
            The rest of the micro-batch within the window of maximum outstanding micro-batch
            backpropagations would checkpoint all Transformer layers.

    Returns:
        a list of loss `torch.Tensor`s if the last stage, empty list otherwise.

    """
    # timers = get_timers()

    if deallocate_pipeline_outputs:
        warnings.warn(
            "`deallocate_pipeline_outputs` is experimental and subject to change. "
            "This option is not recommended."
        )

    model: List[torch.nn.Module] = listify_model(model)
    if len(model) != 1:
        msg = f"`model` is expected be a `nn.Module`, but {type(model)}"
        raise RuntimeError(msg)
    model: torch.nn.Module = model[0]

    # Disable async grad reductions
    if custom_sync_context_handler is not None:
        sync_context_handler = custom_sync_context_handler
    else:
        sync_context_handler = contextlib.nullcontext
    sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal sync_context
        if sync_context is None:
            sync_context = sync_context_handler()
            sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal sync_context
        if sync_context is not None:
            sync_context.__exit__(None, None, None)
            sync_context = None
    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_microbatches: int = get_num_microbatches()
    num_warmup_microbatches: int = (
        parallel_state.get_pipeline_model_parallel_world_size() - parallel_state.get_pipeline_model_parallel_rank() - 1
    )
    num_warmup_microbatches: int = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining: int = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_micro_batches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if num_micro_batches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    rank: int = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes: List[List[int]] = get_tensor_shapes(
        rank - 1,
        model_type,
        tensor_shape=tensor_shape,
        decoder_sequence_length=decoder_sequence_length,
        sequence_parallel_enabled=sequence_parallel_enabled,
    )
    send_tensor_shapes: List[List[int]] = get_tensor_shapes(
        rank,
        model_type,
        tensor_shape=tensor_shape,
        decoder_sequence_length=decoder_sequence_length,
        sequence_parallel_enabled=sequence_parallel_enabled,
    )

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

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_micro_batch = (
                i % max_outstanding_backprops >= num_micro_batches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_micro_batch = None
        input_tensor = recv_forward(
            tensor_shapes=recv_tensor_shapes,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
        )
        cur_microbatch: Optional[torch.Tensor] = get_kth_microbatch(batch, i)
        output_tensor = forward_step(
            forward_step_func,
            cur_microbatch,
            model,
            input_tensor,
            losses_reduced,
            dtype,
            disable_autocast,
            checkpoint_activations_micro_batch,
        )
        _logger.debug("send fwd")
        send_forward(
            output_tensor,
            tensor_shapes=send_tensor_shapes,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
        )

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            free_output_tensor(output_tensor, deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        _logger.debug("recv_forward before steady state start")
        input_tensor: List[Union[None, torch.Tensor, FutureTensor]] = recv_forward(
            tensor_shapes=recv_tensor_shapes,
            dtype=dtype,
            async_comm=async_comm,
            sync_batch_comm=sync_batch_comm,
        )

    ###################################################################################################################
    # Run 1F1B in steady state.
    ###################################################################################################################
    _logger.info("Steady phase")
    for i in range(num_microbatches_remaining):
        _logger.debug(f"steady iter: {i} / {num_microbatches_remaining}")
        last_iteration: bool = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_micro_batch = (
                ((i+num_warmup_microbatches) % max_outstanding_backprops) >= num_micro_batches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_micro_batch = None
        cur_microbatch: Optional[torch.Tensor] = get_kth_microbatch(batch, i + num_warmup_microbatches)
        output_tensor: Union[torch.Tensor, Sequence[torch.Tensor]] = forward_step(
            forward_step_func,
            cur_microbatch,
            model,
            input_tensor,
            losses_reduced,
            dtype,
            disable_autocast,
            checkpoint_activations_micro_batch,
        )
        if forward_only:
            _logger.debug("send fwd")
            send_forward(
                output_tensor,
                tensor_shapes=send_tensor_shapes,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
            )

            if not last_iteration:
                _logger.debug("receive fwd (last iteration)")
                input_tensor = recv_forward(
                    tensor_shapes=recv_tensor_shapes,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                )

        else:
            _logger.debug("send fwd & receive bwd")
            output_tensor_grad = send_forward_recv_backward(
                output_tensor,
                tensor_shapes=send_tensor_shapes,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            free_output_tensor(output_tensor, deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(
                input_tensor,
                output_tensor,
                output_tensor_grad,
                model_type=model_type,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )

            if last_iteration:
                input_tensor = None
                _logger.debug("send bwd")
                send_backward(
                    input_tensor_grad,
                    tensor_shapes=recv_tensor_shapes,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                )
            else:
                _logger.debug("send bwd and receive fwd")
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad,
                    tensor_shapes=recv_tensor_shapes,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                )
    ###################################################################################################################
    # Run cooldown backward passes.
    ###################################################################################################################
    _logger.info("Cooldown phase")
    if not forward_only:
        for i in range(num_warmup_microbatches):
            _logger.debug(f"cooldown iter: {i} / {num_warmup_microbatches}")

            if i == num_warmup_microbatches-1 and rank == 0:
                # Async grad reduction in first pipeline stage, during
                # last backward pass
                enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            _logger.debug("receive bwd")
            output_tensor_grad = recv_backward(
                tensor_shapes=send_tensor_shapes,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
            )

            input_tensor_grad = backward_step(
                input_tensor,
                output_tensor,
                output_tensor_grad,
                model_type=model_type,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )

            _logger.debug("send bwd")
            send_backward(
                input_tensor_grad,
                tensor_shapes=recv_tensor_shapes,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
            )

    # Grad reduction in all pipeline stages except the first, during
    # the bubble overhead
    enable_grad_sync()
    if rank != 0 and custom_grad_sync_func is not None:
        custom_grad_sync_func()

    return losses_reduced
