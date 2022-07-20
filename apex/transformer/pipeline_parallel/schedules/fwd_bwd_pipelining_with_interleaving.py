from typing import List, Union, Optional, Sequence
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

    _logger.info(
        f"num_microbatches: {num_microbatches}, "
        f"num_warmup_microbatches: {num_warmup_microbatches}, "
        f"num_microbatches_remaining: {num_microbatches_remaining}"
    )

    ###################################################################################################################
    # Helper function definitions.
    ###################################################################################################################
    def get_model_chunk_id(microbatch_id: int, forward: bool) -> int:
        """Helper function to get the model chunk ID given the iteration number."""
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        microbatch_id_in_group = microbatch_id % (
            pipeline_parallel_size * num_model_chunks
        )
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def forward_step_helper(microbatch_id: int, curr_iters: List[int]) -> torch.Tensor:
        """Helper method to run forward step with model split into chunks

        (run set_virtual_pipeline_model_parallel_rank() before calling forward_step()).
        """
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

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

        return input_tensor_grad

    ###################################################################################################################
    # Run warmup forward passes.
    ###################################################################################################################
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(
            tensor_shape=tensor_shape,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )
    )
    _logger.info("Warmup phase")
    for k in range(num_warmup_microbatches):
        _logger.debug(f"warmup iter: {k} / {num_warmup_microbatches}")
        output_tensor = forward_step_helper(k, curr_iters)

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
        output_tensor = forward_step_helper(forward_k, curr_iters)

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
        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
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
                )
            )

    return losses_reduced
