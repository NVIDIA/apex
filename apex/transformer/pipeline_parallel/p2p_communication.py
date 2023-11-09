# coding=utf-8
# Copyright (c) 2021-22, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO(mkozuki): Consider removing `timers`.

from functools import reduce
import operator
from typing import Union, Optional, Tuple

import torch

from apex.transformer import parallel_state
from apex.transformer.log_util import get_transformer_logger
from apex.transformer.utils import split_tensor_into_1d_equal_chunks
from apex.transformer.utils import gather_split_1d_tensor
from apex.transformer.pipeline_parallel.utils import Shape
from apex.transformer.pipeline_parallel._timers import _Timers


_logger = get_transformer_logger(__name__)


class FutureTensor:
    def __init__(self, tensor: torch.Tensor, waitfunc):
        self.tensor = tensor
        self.waitfunc = waitfunc

    def get(self):
        if self.waitfunc is not None:
            res = self.waitfunc()
            if isinstance(res, torch.Tensor):
                self.tensor = res
            self.waitfunc = None
        return self.tensor


def _run_p2pops(
    tensor_send_prev: Union[torch.Tensor, None],
    tensor_send_next: Union[torch.Tensor, None],
    tensor_recv_prev: Union[torch.Tensor, None],
    tensor_recv_next: Union[torch.Tensor, None],
    async_comm: bool = False,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
):
    p2p_group = parallel_state.get_pipeline_model_parallel_group()
    default_group = parallel_state.get_model_parallel_group()

    need_to_sync = p2p_group.name() != default_group.name()
    reqs = []
    ops = []

    if batch_p2p_comm and p2p_group.name() == "nccl":
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                op=torch.distributed.isend,
                tensor=tensor_send_prev,
                peer=parallel_state.get_pipeline_model_parallel_prev_rank(),
                group=p2p_group,
            )
            ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                op=torch.distributed.irecv,
                tensor=tensor_recv_prev,
                peer=parallel_state.get_pipeline_model_parallel_prev_rank(),
                group=p2p_group,
            )
            ops.append(recv_prev_op)
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                op=torch.distributed.isend,
                tensor=tensor_send_next,
                peer=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=p2p_group,
            )
            ops.append(send_next_op)
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                op=torch.distributed.irecv,
                tensor=tensor_recv_next,
                peer=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=p2p_group,
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            # sync before communication if needed
            if need_to_sync:
                torch.cuda.synchronize()
            reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        # sync before communication if needed
        if need_to_sync and any([
            tensor_send_prev is not None, tensor_recv_prev is not None,
            tensor_send_next is not None, tensor_recv_next is not None]):
            torch.cuda.synchronize()

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev,
                dst=parallel_state.get_pipeline_model_parallel_prev_rank(),
                group=p2p_group,
            )
            reqs.append(send_prev_req)
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev,
                src=parallel_state.get_pipeline_model_parallel_prev_rank(),
                group=p2p_group,
            )
            reqs.append(recv_prev_req)
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next,
                dst=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=p2p_group,
            )
            reqs.append(send_next_req)        
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.irecv(
                tensor=tensor_recv_next,
                src=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=p2p_group,
            )
            reqs.append(recv_next_op)

    if len(reqs) > 0:
        if overlap_p2p_comm:
            return (None, None, None, None, reqs)

        if async_comm:
            if len(ops) == 0 or len(reqs) == len(ops):
                tensor_send_prev_req = None if tensor_send_prev is None else reqs.pop(0)
                tensor_recv_prev_req = None if tensor_recv_prev is None else reqs.pop(0)
                tensor_send_next_req = None if tensor_send_next is None else reqs.pop(0)
                tensor_recv_next_req = None if tensor_recv_next is None else reqs.pop(0)
            elif len(reqs) == 1:
                tensor_send_prev_req = None if tensor_send_prev is None else reqs[0]
                tensor_recv_prev_req = None if tensor_recv_prev is None else reqs[0]
                tensor_send_next_req = None if tensor_send_next is None else reqs[0]
                tensor_recv_next_req = None if tensor_recv_next is None else reqs[0]
            else:
                assert False, "failed to manage p2p requests and handles"
            return (tensor_send_prev_req, tensor_recv_prev_req, tensor_send_next_req, tensor_recv_next_req, None)
        else:
            for req in reqs:
                req.wait()
            return (None, None, None, None, None)
    return (None, None, None, None, None)


# TODO(mkozuki): Check if it's possible to sunset `override_scatter_gather_tensors_in_pipeline`.
# TODO(mkozuki): Think about if it's possible to push some logic and arguments e.g.
# `scatter_gather_tensors_in_pipeline`, `sequence_parallel_enabled`, and
# `override_scatter_gather_tensors_in_pipeline` # to the user of
# apex.transformer forward_backwardfunctions.
def _communicate(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Optional[Shape] = None,
    override_scatter_gather_tensors_in_pipeline: bool = False,
    dtype_: Optional[torch.dtype] = None,
    *,
    scatter_gather_tensors_in_pipeline: bool = True,
    params_dtype: Optional[torch.dtype] = None,
    fp32_residual_connection: bool = False,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
) -> Tuple[Union[torch.Tensor, FutureTensor, None], Union[torch.Tensor, FutureTensor, None]]:
    """Base function for communication of tensors between stages.


    .. note::
        Reference https://github.com/NVIDIA/Megatron-LM/blob/cfd2e2160700b7f2c1bf35298ac14bc341f4c759/megatron/p2p_communication.py#L24-L159

    dtype logic: If none of ``dtype_``, ``params_dtype``, ``fp32_residual_connection`` is specified,
    torch.float32 is used.

    See https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/arguments.py#L145-L159
    for the details of arguments of ``dtype_``, ``params_dtype``, ``fp32_residual_connection``.

    Args:
        tensor_send_next: tensor to send to next rank (no tensor sent if set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if set to None).
        recv_prev: boolean for whether tensor should be received from previous rank.
        recv_next: boolean for whether tensor should be received from next rank.
        tensor_shape: optional, use when the input sequence contains less tokens than the default sequence length
        override_scatter_gather_tensors_in_pipeline:
            optional, this is used when tensor_shape is provided to override scatter gather tensors
        dtype_: This is used when tensor_shape is provided and what is the type of tensor_shape

    Keyword args:
        scatter_gather_tensors_in_pipeline: Optional. If :obj:`True`, use scatter/gather to optimize communication of tensors.
        params_dtype: Optional and legacy. Defaults to torch.float. If you manually call `.half()` or `.bfloat16()` on
            your model deliberately, pass this argument.
        fp32_residual_connection: Optional. If :obj:`True`, move residual connections to fp32.
        sequence_parallel_enabled: Set to :obj:`True` if sequence parallel is enabled.
            This argument is here for consistency with Megatron-LM.
            This argument has an effect on the communication optimization, not on tensor_shape update.
        sync_batch_comm: If :obj:`False`, disable cuda synchronization after the batched communication.
            To disable, https://github.com/pytorch/pytorch/pull/82450 would be required.
        overlap_p2p_comm: If :obj:`True`, returns cuda wait handles to scheduler instead of completing
            the communication within the p2p transfer API instance. The scheduler manages the communication completion
            to overlap with computation.
        batch_p2p_comm: If :obj:`True`, use the batched send and receive api to conduct the communication of
            a collection of send and receive operations between peer. If :obj:`False`, conduct each send and recv operation
            individually.

    Returns:
        tuple containing

        - tensor_recv_prev: `torch.Tensor` if `recv_prev` is :obj:`True`, `None` otherwise.
        - tensor_recv_next: `torch.Tensor` if `recv_next` is :obj:`True`, `None` otherwise.
    """
    if async_comm and sequence_parallel_enabled:
        import warnings  # NOQA
        class ExperimentalWarning(UserWarning): pass  # NOQA
        warnings.warn(
            "The combination of `async_comm` and `sequence_parallel_enabled` is not well tested.",
            ExperimentalWarning,
        )
    # Create placeholder tensors for receive in forward and backward directions if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    if tensor_shape is None:
        # In megatron, `tensor_shape` is set to `(args.seq_length, args.micro_batch_size, args.hidden_size)`
        raise RuntimeError(
            "`tensor_shape` must be specified. Common `tensor_shape` is `(seq_length, micro_batch_size, hidden_size)`")

    tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
    override_scatter_gather_tensors_in_pipeline_ = False
    # TODO(mkozuki): Demystify hardcode False of `scatter_gather_tensors_in_pipeline` and add a testcase if possible.
    # NOTE(mkozuki): This is super strange and doesn't make sense to me. I have no idea what is happening here.
    # However, I can say that this hardcoding override is necessary for sequence parallel in nemo megatron to work.
    # I've not managed to reproduce the hang using standalone GPT with sequence parallel.
    # The hang in NeMo Megatron happens in the 3rd iteration, the last iteration of stead phase inside
    # forward_backward_pipelining_without_interleaving, pipeline parallel rank of 0 (tensor model parallel world
    # size of 2 and pipeline model parallel world size of 2). The commit then of APEX and NeMo were
    # https://github.com/NVIDIA/apex/pull/1396/commits/3060c98dd8ba42abf7702ea9d2cff0f39ea74f45 and
    # https://github.com/NVIDIA/NeMo/pull/4232/commits/1cb32dfca2ab9b20f53ebdb84476c34cb42f0205.
    # The PyTorch version was 1.13.0a0+git2d354cd, for what is worth.
    # Currently, indiscriminately this is set to `False`, which can lead to an unexpected performance regression
    # for non sequence parallel case.
    scatter_gather_tensors_in_pipeline = False
    if scatter_gather_tensors_in_pipeline and not sequence_parallel_enabled:
        tensor_chunk_size = int(reduce(operator.mul, tensor_shape, 1))
        if tensor_chunk_size % tensor_parallel_size == 0:
            tensor_chunk_shape = [tensor_chunk_size // tensor_parallel_size]
        else:
            tensor_chunk_shape = tensor_shape
            override_scatter_gather_tensors_in_pipeline_ = True
    else:
        tensor_chunk_shape = tensor_shape

    # The dtype logic below is copied from NVIDIA/Megatron-LM repo:
    # https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/p2p_communication.py#L74-L81
    dtype = params_dtype or torch.float
    if fp32_residual_connection:
        dtype = torch.float
    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        # TODO(mkozuki): Figure out why this logic of requires_grad isn't working
        # when sequence_parallel_enabled=True. Otherwise, `x.retain_grad()` of
        # https://github.com/crcrpar/apex/blob/069832078a652b4bd8a99db84faf953a81415ab3/apex/transformer/pipeline_parallel/schedules/common.py#L360
        # fails.
        # requires_grad = False

    if recv_prev:
        tensor_recv_prev = torch.empty(
            tensor_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    if recv_next:
        tensor_recv_next = torch.empty(
            tensor_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )

    # Split tensor into smaller chunks if using scatter-gather optimization.
    scatter_gather_optimization_doable = (
        not override_scatter_gather_tensors_in_pipeline_
        and scatter_gather_tensors_in_pipeline
        and not sequence_parallel_enabled
    )
    if scatter_gather_optimization_doable:
        if tensor_send_next is not None:
            tensor_send_next = split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = split_tensor_into_1d_equal_chunks(tensor_send_prev)

    # Send tensors in both the forward and backward directions as appropriate.
    tensor_send_prev_req, tensor_recv_prev_req, tensor_send_next_req, tensor_recv_next_req, wait_handles = _run_p2pops(
        tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next, async_comm, overlap_p2p_comm, batch_p2p_comm)

    if async_comm:
        tensor_recv_prev_waitfunc = None
        tensor_recv_next_waitfunc = None
        # TODO: investigate whether this is necessary for correctness (ref: https://github.com/pytorch/pytorch/issues/38642)
        # see also: sync added for async_comm callbacks below in gather_recv_prev_wait and gather_recv_next_wait
        if tensor_recv_prev_req is not None:
            def tensor_recv_prev_wait():
                tensor_recv_prev_req.wait()
                torch.cuda.synchronize()
            tensor_recv_prev_waitfunc = tensor_recv_prev_wait
        if tensor_recv_next_req is not None:
            def tensor_recv_next_wait():
                tensor_recv_next_req.wait()
                torch.cuda.synchronize()
            tensor_recv_next_waitfunc = tensor_recv_next_wait
    else:
        if sync_batch_comm:
            # To protect against race condition when using batch_isend_irecv().
            torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    if scatter_gather_optimization_doable:
        if not async_comm:
            if recv_prev:
                tensor_recv_prev = (
                    gather_split_1d_tensor(tensor_recv_prev)
                    .view(tensor_shape)
                    .requires_grad_()
                )

            if recv_next:
                tensor_recv_next = (
                    gather_split_1d_tensor(tensor_recv_next)
                    .view(tensor_shape)
                    .requires_grad_()
                )
        else:
            def gather_recv_prev_wait():
                tensor_recv_prev_req.wait()
                # From @Deepak's PR https://github.com/NVIDIA/Megatron-LM/commit/27fc468964064eeb33b703c9a0b2af938d80dd14
                # A sync seems to be needed before gather otherwise losses jump around e.g., in run_gpt_minimal_test
                torch.cuda.synchronize()
                return (
                    gather_split_1d_tensor(tensor_recv_prev)
                    .view(tensor_shape)
                    .requires_grad_()
                )
            def gather_recv_next_wait():
                tensor_recv_next_req.wait()
                torch.cuda.synchronize()
                return (
                    gather_split_1d_tensor(tensor_recv_next)
                    .view(tensor_shape)
                    .requires_grad_()
                )
            tensor_recv_prev_waitfunc = gather_recv_prev_wait
            tensor_recv_next_waitfunc = gather_recv_next_wait
    if async_comm:
        future_tensor_recv_prev = None
        future_tensor_recv_next = None
        if tensor_recv_prev is not None:
            future_tensor_recv_prev = FutureTensor(tensor_recv_prev, tensor_recv_prev_waitfunc)
        if tensor_recv_next is not None:
            future_tensor_recv_next = FutureTensor(tensor_recv_next, tensor_recv_next_waitfunc)
        return future_tensor_recv_prev, future_tensor_recv_next, None
    return tensor_recv_prev, tensor_recv_next, wait_handles


def recv_forward(
    tensor_shape: Shape,
    override_scatter_gather_tensors_in_pipeline: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Receive tensor from previous rank in pipeline (forward receive)."""
    if parallel_state.is_pipeline_first_stage():
        return None
    # if timers is not None:
    #     timers("forward-recv").start()
    input_tensor, _, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("forward-recv").stop()
    return input_tensor


def recv_backward(
    tensor_shape: Shape = None,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Receive tensor from next rank in pipeline (backward receive)."""
    if parallel_state.is_pipeline_last_stage():
        return None
    # if timers is not None:
    #     timers("backward-recv").start()
    _, output_tensor_grad, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("backward-recv").stop()
    return output_tensor_grad


def send_forward(
    output_tensor: torch.Tensor,
    override_scatter_gather_tensors_in_pipeline: bool = False,
    tensor_shape: Shape = None,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> None:
    """Send tensor to next rank in pipeline (forward send)."""
    if parallel_state.is_pipeline_last_stage():
        return
    # if timers is not None:
    #     timers("forward-send").start()
    _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("forward-send").stop()


def send_backward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> None:
    """Send tensor to previous rank in pipeline (backward send)."""
    if parallel_state.is_pipeline_first_stage():
        return
    # if timers is not None:
    #     timers("backward-send").start()
    _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("backward-send").stop()


def send_forward_recv_backward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Batched send and recv with next rank in pipeline."""
    if parallel_state.is_pipeline_last_stage():
        return None
    # if timers is not None:
    #     timers("forward-send-backward-recv").start()
    _, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("forward-send-backward-recv").stop()
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> Union[torch.Tensor, FutureTensor, None]:
    """Batched send and recv with previous rank in pipeline."""
    if parallel_state.is_pipeline_first_stage():
        return None
    # if timers is not None:
    #     timers("backward-send-forward-recv").start()
    input_tensor, _, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("backward-send-forward-recv").stop()
    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> Union[torch.Tensor, FutureTensor]:
    """Batched recv from previous rank and send to next rank in pipeline."""
    # if timers is not None:
    #     timers("forward-send-forward-recv").start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("forward-send-forward-recv").stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> Union[torch.Tensor, FutureTensor]:
    """Batched recv from next rank and send to previous rank in pipeline."""
    # if timers is not None:
    #     timers("backward-send-backward-recv").start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("backward-send-backward-recv").stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    *,
    dtype: Optional[torch.dtype] = None,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    sync_batch_comm: bool = True,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    timers: _Timers = None,
) -> Tuple[Union[torch.Tensor, FutureTensor], Union[torch.Tensor, FutureTensor]]:
    """Batched send and recv with previous and next ranks in pipeline."""
    # if timers is not None:
    #     timers("forward-backward-send-forward-backward-recv").start()
    input_tensor, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype_=dtype,
        async_comm=async_comm,
        sequence_parallel_enabled=sequence_parallel_enabled,
        sync_batch_comm=sync_batch_comm,
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
    )
    # if timers is not None:
    #     timers("forward-backward-send-forward-backward-recv").stop()
    if overlap_p2p_comm:
        return input_tensor, output_tensor_grad, wait_handles
    return input_tensor, output_tensor_grad
