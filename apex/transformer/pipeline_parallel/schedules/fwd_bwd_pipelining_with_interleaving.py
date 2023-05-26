import contextlib
from typing import Any, Callable, List, Optional, Sequence, Union
import warnings

import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.pipeline_parallel.schedules.common import Batch
from apex.transformer.pipeline_parallel.schedules.common import FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import backward_step
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import free_output_tensor
from apex.transformer.pipeline_parallel.utils import get_kth_microbatch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import get_model_type
from apex.transformer.log_util import get_transformer_logger


__all__ = ["_forward_backward_pipelining_with_interleaving"]


_logger = get_transformer_logger(__name__)


# TODO(mkozuki): Reduce cyclomatic complexity
def _forward_backward_pipelining_with_interleaving(
    forward_step_func: FwdStepFunc,
    batch: List[Optional[Batch]],
    model: List[torch.nn.Module],
    *,
    forward_only: bool,
    tensor_shape: Optional[Union[List[int], torch.Size]] = None,
    dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    disable_autocast: bool = False,
    deallocate_pipeline_outputs: bool = False,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    custom_sync_context_handler: Optional[Callable] = None,
    custom_grad_sync_func: Optional[Callable] = None,
    custom_param_sync_func: Optional[Callable] = None,
    sync_batch_comm: bool = True,
    num_micro_batches_with_partial_activation_checkpoints: Optional[int] = None,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    **kwargs,
) -> List[Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Run interleaved 1F1B schedule with communication between pipeline stages as needed.

    This function assumes `batch` and `model` is a list of `Batch`'s and a list of `torch.nn.Module`, respectively.
    This means that model is split into model chunks.

    This pipeline parallel scheduling consists of three steps:
        1. warmup
        2. 1F1B a.k.a. steady state
        3. cooldown
    Note that if `forward_only` this scheduling consists of only warmup phase.

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
        custom_sync_context_handler: If provided, this is treated as a
            function to construct a context manager to disable
            asynchronous gradient reductions. Asynchronous gradient
            reductions are only enabled in the final backward pass of
            each model chunk.
        custom_grad_sync_func: If provided, this is treated as a
            function to launch asynchronous gradient reductions (e.g.
            reduce-scatters with distributed optimizer). The function
            should take one positional argument: a list of parameters
            whose gradients should be synchronized. Asynchronous
            gradient reductions are launched after the final backward
            pass of each model chunk.
        custom_param_sync_func: If provided, this is treated as a
            function to launch asynchronous parameter synchronizations
            (e.g. all-gathers with distributed optimizer). The
            function should take one positional argument: a list of
            parameters whose values should be synchronized.
            Asynchronous parameter synchronizations are launched
            before the first forward pass of each model chunk.
        sync_batch_comm: If :obj:`False`, disable cuda synchronization after the batched communication.
            To disable, https://github.com/pytorch/pytorch/pull/82450 would be required.
        num_micro_batches_with_partial_activation_checkpoints: If :obj:`int`, set the number of
            micro-batches checkpointing the activation of partial number of Transformer layers.
            The rest of the micro-batch within the window of maximum outstanding micro-batch
            backpropagations would checkpoint all Transformer layers.
        overlap_p2p_comm: If :obj:`True`, returns cuda wait handles to scheduler instead of completing
            the communication within the p2p transfer API instance. The scheduler manages the communication completion
            to overlap with computation.
        batch_p2p_comm: If :obj:`True`, use the batched send and receive api to conduct the communication of
            a collection of send and receive operations between peer. If :obj:`False`, conduct each send and recv operation
            individually.

    Returns:
        a list of loss `torch.Tensor`s if the last stage, empty list otherwise.

    """
    if not isinstance(model, list):
        raise RuntimeError("`model` must be a list of `nn.Module`'s'")

    if deallocate_pipeline_outputs:
        warnings.warn(
            "`deallocate_pipeline_outputs` is experimental and subject to change. "
            "This option is not recommended."
        )

    # Construct helper functions for async grad reductions
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

    # mypy will blame the following if statement
    if sequence_parallel_enabled:
        seq_length, batch_size, hidden = tensor_shape
        tensor_shape = (
            seq_length // parallel_state.get_tensor_model_parallel_world_size(),
            batch_size,
            hidden,
        )

    num_model_chunks: int = len(model)
    input_tensors: List[List[Union[None, torch.Tensor]]] = [
        [] for _ in range(num_model_chunks)
    ]
    output_tensors: List[List[Union[None, torch.Tensor]]] = [
        [] for _ in range(num_model_chunks)
    ]
    curr_iters: List[int] = [0 for _ in range(num_model_chunks)]
    losses_reduced: List[Union[None, torch.Tensor]] = []
    if not forward_only:
        output_tensor_grads: List[List[Union[None, torch.Tensor]]] = [
            [] for _ in range(num_model_chunks)
        ]

    pipeline_parallel_size: int = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank: int = parallel_state.get_pipeline_model_parallel_rank()

    # Compute number of warmup and remaining microbatches.
    num_microbatches: int = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches: bool = False
    if forward_only:
        num_warmup_microbatches: int = num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (
                pipeline_parallel_size - pipeline_parallel_rank - 1
            ) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
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

    _logger.info(
        f"num_microbatches: {num_microbatches}, "
        f"num_warmup_microbatches: {num_warmup_microbatches}, "
        f"num_microbatches_remaining: {num_microbatches_remaining}"
    )

    # Synchronize params for first two model chunks
    if custom_param_sync_func is not None:
        custom_param_sync_func(model[0].parameters())
        custom_param_sync_func(model[1].parameters())

    ###################################################################################################################
    # Helper function definitions.
    ###################################################################################################################
    def get_model_chunk_id(microbatch_id: int, forward: bool) -> int:
        """Helper function to get the model chunk ID given the iteration number.

        Each model chunk processes pipeline_parallel_size microbatches
        at a time. We assume that the number of microbatches is a
        multiple of pipeline_parallel_size*num_model_chunks.
        """
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Helper function to check if an iteration is the first for a model
        chunk.
        """
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Helper function to check if an iteration is the last for a model
        chunk.
        """
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(
        microbatch_id: int,
        curr_iters: List[int],
        checkpoint_activations_micro_batch: Optional[bool] = None,
        ) -> torch.Tensor:
        """Helper method to run forward step with model split into chunks

        (run set_virtual_pipeline_model_parallel_rank() before calling forward_step()).
        """
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: To achieve maximum performance, pipeline parallelism
        # assumes all ranks have the same compute time. However,
        # asynchronous communication tends to slow down compute. Thus,
        # we launch asynchronous communication at the same time across
        # the pipeline-parallel group.
        if custom_param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if param_sync_microbatch_id < num_microbatches and is_first_microbatch_for_model_chunk(param_sync_microbatch_id):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    custom_param_sync_func(model[param_sync_chunk_id].parameters())

        # forward step
        if parallel_state.is_pipeline_first_stage() and len(
            input_tensors[model_chunk_id]
        ) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(
            forward_step_func,
            get_kth_microbatch(batch, curr_iters[model_chunk_id]),
            model[model_chunk_id],
            input_tensor,
            losses_reduced,
            dtype,
            disable_autocast,
            checkpoint_activations_micro_batch,
        )
        curr_iters[model_chunk_id] += 1
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id: int) -> torch.Tensor:
        """Helper method to run backward step with model split into chunks

        (run set_virtual_pipeline_model_parallel_rank() before calling backward_step()).
        """
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        model_type = get_model_type(model[model_chunk_id])
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if custom_grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
                enable_grad_sync()

        # backward step
        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor,
            output_tensor,
            output_tensor_grad,
            model_type=model_type,
            grad_scaler=grad_scaler,
            deallocate_pipeline_outputs=deallocate_pipeline_outputs,
        )

        # launch grad synchronization (custom grad sync)
        # Note: To achieve maximum performance, pipeline parallelism
        # assumes all ranks have the same compute time. However,
        # asynchronous communication tends to slow down compute. Thus,
        # we launch asynchronous communication at the same time across
        # the pipeline-parallel group.
        if custom_grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(grad_sync_microbatch_id):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                custom_grad_sync_func(model[grad_sync_chunk_id].parameters())
        disable_grad_sync()

        return input_tensor_grad

    ###################################################################################################################
    # Run warmup forward passes.
    ###################################################################################################################
    fwd_wait_handles, bwd_wait_handles = None, None
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(
            tensor_shape=tensor_shape,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
            batch_p2p_comm=batch_p2p_comm,
        )
    )
    _logger.info("Warmup phase")
    for k in range(num_warmup_microbatches):
        _logger.debug(f"warmup iter: {k} / {num_warmup_microbatches}")

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_micro_batch = k % max_outstanding_backprops >= \
                num_micro_batches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_micro_batch = None

        if fwd_wait_handles is not None:
            for wait_handle in fwd_wait_handles:
                wait_handle.wait()

        output_tensor = forward_step_helper(k, curr_iters, checkpoint_activations_micro_batch)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False
        _logger.debug(
            f"next fwd model chunk ID: {next_forward_model_chunk_id}, recv_prev: {recv_prev}"
        )

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            _logger.debug("Pipeline last stage, not sending tensor downstream")
            output_tensor = None

        if overlap_p2p_comm:
            # P2P communications in warmup are not overlapped with computes. We split P2P
            # communications for activation forward and activation_gradient backward in warmup,
            # to match the send/recv API granularity in 1F1B in case of using batched send/recv API.

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            _logger.debug("send fwd and receive fwd")
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                overlap_p2p_comm=True,
                batch_p2p_comm=batch_p2p_comm,
            )
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                _logger.debug("send bwd and receive bwd")
                output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    overlap_p2p_comm=True,
                    batch_p2p_comm=batch_p2p_comm,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                _logger.debug("send fwd&bwd and receive fwd&bwd")
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                _logger.debug("send fwd and receive fwd")
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
            free_output_tensor(output_tensor, deallocate_pipeline_outputs)

    ###################################################################################################################
    # Run 1F1B in steady state.
    ###################################################################################################################
    _logger.info("Steady phase")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        _logger.debug(f" steady phase iter {k} / {num_microbatches_remaining}")
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_micro_batch = (
                forward_k % max_outstanding_backprops >= num_micro_batches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_micro_batch = None

        if overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for wait_handle in fwd_wait_handles:
                    wait_handle.wait()

            output_tensor = forward_step_helper(forward_k, curr_iters, checkpoint_activations_micro_batch)

            # Set forward model chunk id
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if the current virtual stage has an activation tensor to receive
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k + 1, forward=True
                )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the 
            # previous stage
            _logger.debug("send fwd and receive fwd")
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                overlap_p2p_comm=True,
                batch_p2p_comm=batch_p2p_comm,
            )

            if bwd_wait_handles is not None:
                for wait_handle in bwd_wait_handles:
                    wait_handle.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Set backward model chunk id
            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            _logger.debug(
                f"fwd/bwd model chunk id: {forward_model_chunk_id}/{backward_model_chunk_id}"
            )

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k + 1, forward=False
                )

            # Send activation grad tensor to the previous stage and receive activation grad tensor
            # from the previous stage
            _logger.debug("send bwd and receive bwd")
            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                overlap_p2p_comm=True,
                batch_p2p_comm=batch_p2p_comm,
            )
        else:
            output_tensor = forward_step_helper(forward_k, curr_iters, checkpoint_activations_micro_batch)

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            _logger.debug(
                f"fwd/bwd model chunk id: {forward_model_chunk_id}/{backward_model_chunk_id}"
            )
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k + 1, forward=True
                )

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k + 1, forward=False
                )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            _logger.debug("send fwd&bwd and receive fwd&bwd")
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                batch_p2p_comm=batch_p2p_comm,
            )
            free_output_tensor(output_tensor, deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    ###################################################################################################################
    # Run cooldown backward passes (flush out pipeline).
    ###################################################################################################################
    _logger.info("Cooldown phase")
    if not forward_only:
        if overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
            )

        for k in range(num_microbatches_remaining, num_microbatches):
            _logger.debug(
                f"cooldown iter {k} in range({num_microbatches_remaining}, {num_microbatches})"
            )
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False

            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
            )

    # Make sure to exit context handler for async grad reductions
    enable_grad_sync()

    return losses_reduced
