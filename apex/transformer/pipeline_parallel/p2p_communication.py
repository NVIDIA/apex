# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from functools import reduce
import operator
from typing import Union, Optional, Tuple

import torch

from apex.transformer import parallel_state
from apex.transformer.utils import split_tensor_into_1d_equal_chunks
from apex.transformer.utils import gather_split_1d_tensor
from apex.transformer.pipeline_parallel.utils import Shape
from apex.transformer.pipeline_parallel._timers import _Timers


def _run_p2pops(
        tensor_send_prev: Union[torch.Tensor, None],
        tensor_send_next: Union[torch.Tensor, None],
        tensor_recv_prev: Union[torch.Tensor, None],
        tensor_recv_next: Union[torch.Tensor, None],
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            parallel_state.get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            parallel_state.get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            parallel_state.get_pipeline_model_parallel_next_rank(),
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            parallel_state.get_pipeline_model_parallel_next_rank(),
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()


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
) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    """Base function for communication of tensors between stages.

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

    Returns:
        tuple containing

        - tensor_recv_prev: `torch.Tensor` if `recv_prev` is :obj:`True`, `None` otherwise.
        - tensor_recv_next: `torch.Tensor` if `recv_next` is :obj:`True`, `None` otherwise.
    """
    # Create placeholder tensors for receive in forward and backward directions if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    if tensor_shape is None:
        # In megatron, `tensor_shape` is set to `(args.seq_length, args.micro_batch_size, args.hidden_size)`
        raise RuntimeError(
            "`tensor_shape` must be specified. Common `tensor_shape` is `(seq_length, micro_batch_size, hidden_size)`")
    if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
        tensor_chunk_shape = (reduce(operator.mul, tensor_shape, 1) // parallel_state.get_tensor_model_parallel_world_size(),)
    else:
        tensor_chunk_shape = tensor_shape

    # The dtype logic below is copied from NVIDIA/Megatron-LM repo:
    # https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/p2p_communication.py#L74-L81
    # NOTE (mkozuki): Currently NeMo is implementing APEX AMP O2 style using PyTorch. In O2 style, forcing p2p comm to
    # use FP32 will be a perf killer so that I decided to reanimate `dtype_` argument with the default value of `None`.
    # NOTE (mkozuki): In PyTorch AMP, i.e. `torch.cuda.amp.autocast` context, activation tensors can be either FP32,
    # FP16, or BF16 and there's no way to tell the dtypes of tensors on different devices in general.
    # It might be possible if we restrict model architecture.
    dtype = params_dtype or torch.float
    if fp32_residual_connection:
        dtype = torch.float
    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

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
    if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
        if tensor_send_next is not None:
            tensor_send_next = split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = split_tensor_into_1d_equal_chunks(tensor_send_prev)

    # Send tensors in both the forward and backward directions as appropriate.
    _run_p2pops(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
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

    return tensor_recv_prev, tensor_recv_next


def recv_forward(
        tensor_shape: Shape,
        override_scatter_gather_tensors_in_pipeline: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        timers: _Timers = None,
) -> torch.Tensor:
    """Receive tensor from previous rank in pipeline (forward receive)."""
    if parallel_state.is_pipeline_first_stage():
        return None
    # if timers is not None:
    #     timers("forward-recv").start()
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
        dtype_=dtype,
    )
    # if timers is not None:
    #     timers("forward-recv").stop()
    return input_tensor


def recv_backward(
        tensor_shape: Shape = None,
        *,
        dtype: Optional[torch.dtype] = None,
        timers: _Timers = None,
) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive)."""
    if parallel_state.is_pipeline_last_stage():
        return None
    # if timers is not None:
    #     timers("backward-recv").start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype_=dtype,
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
    )
    # if timers is not None:
    #     timers("forward-send").stop()


def send_backward(
        input_tensor_grad: torch.Tensor,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
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
    )
    # if timers is not None:
    #     timers("backward-send").stop()


def send_forward_recv_backward(
        output_tensor: torch.Tensor,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
        timers: _Timers = None,
) -> Union[None, torch.Tensor]:
    """Batched send and recv with next rank in pipeline."""
    if parallel_state.is_pipeline_last_stage():
        return None
    # if timers is not None:
    #     timers("forward-send-backward-recv").start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        dtype_=dtype,
    )
    # if timers is not None:
    #     timers("forward-send-backward-recv").stop()
    return output_tensor_grad


def send_backward_recv_forward(
        input_tensor_grad: torch.Tensor,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
        timers: _Timers = None,
) -> Union[None, torch.Tensor]:
    """Batched send and recv with previous rank in pipeline."""
    if parallel_state.is_pipeline_first_stage():
        return None
    # if timers is not None:
    #     timers("backward-send-forward-recv").start()
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
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
        timers: _Timers = None,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline."""
    # if timers is not None:
    #     timers("forward-send-forward-recv").start()
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
    )
    # if timers is not None:
    #     timers("forward-send-forward-recv").stop()
    return input_tensor


def send_backward_recv_backward(
        input_tensor_grad: torch.Tensor,
        recv_next: bool,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
        timers: _Timers = None,
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline."""
    # if timers is not None:
    #     timers("backward-send-backward-recv").start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype_=dtype,
    )
    # if timers is not None:
    #     timers("backward-send-backward-recv").stop()
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
        timers: _Timers = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched send and recv with previous and next ranks in pipeline."""
    # if timers is not None:
    #     timers("forward-backward-send-forward-backward-recv").start()
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype_=dtype,
    )
    # if timers is not None:
    #     timers("forward-backward-send-forward-backward-recv").stop()
    return input_tensor, output_tensor_grad
