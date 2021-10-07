# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Callable, Any

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from .. import parallel_state
from .utils import get_num_microbatches
from .utils import unwrap_model
from .utils import get_timers
from .timers import Timers
from . import p2p_communication


def get_forward_backward_func(
    virtual_pipeline_model_parallel_size,
    pipeline_model_parallel_size,
):
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % pipeline_model_parallel_size == 0, (
                "number of microbatches is not divisible by pipeline-parallel "
                "size when using interleaved schedule"
            )
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def forward_step(forward_step_func, data_iterator, model, input_tensor, losses_reduced):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    timers = get_timers()

    timers("forward-compute").start()
    unwrapped_model = unwrap_model(model)
    # NOTE (mkozuki): The passed `model` is expected to implement `set_input_tensor`.
    # See https://github.com/NVIDIA/Megatron-LM/blob/5ac5571ba0265af4c491ee0af1508ca7589450c6/megatron/model/transformer.py#L679  # NOQA
    # for the details of `set_input_tensor`.
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor, loss_func = forward_step_func(data_iterator, model)
    if parallel_state.is_pipeline_last_stage():
        output_tensor = loss_func(output_tensor)
        loss, loss_reduced = output_tensor
        output_tensor = loss / get_num_microbatches()
        losses_reduced.append(loss_reduced)
    timers("forward-compute").stop()

    return output_tensor


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage).
    """

    timers = get_timers()
    timers("backward-compute").start()

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    # Backward pass.
    if output_tensor_grad is None:
        output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad

    timers("backward-compute").stop()

    return input_tensor_grad


@contextmanager
def placeholder_handler():
    try:
        yield
    finally:
        pass


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
    if isinstance(model, DDP):
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


def get_model_chunk_id(microbatch_id: int, num_model_chunks: int, *, forward: bool) -> int:
    """Helper function to get the model chunk ID given the iteration number."""
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
    model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
    if not forward:
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id


@dataclass
class PipeliningWithInterleaving:
    """Class to run interleaved 1F1B schedule where the model is split into model chunks.

    Communication occurs between pipeline stages as needed."""

    forward_step_func: Callable
    data_iterator: Any
    model: torch.nn.Module
    optimizer: torch.optim.optimizer.Optimizer
    timers: Timers
    forward_only: bool

    def __post_init__(self):
        self.num_model_chunks = len(self.model)
        self.num_microbatches = get_num_microbatches() * self.num_model_chunks
        self.pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.input_tensors, self.output_tensors = [[[] for _ in range(self.num_model_chunks)] for _ in range(2)]
        self.losses_reduced = []
        if not self.forward_only:
            self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]
        all_warmup_microbatches = False
        if self.forward_only:
            num_warmup_microbatches = self.num_microbatches
        else:
            # Run all forward passes and then all backward passes if number of
            # microbatches is just the number of pipeline stages.
            # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            if get_num_microbatches() == self.pipeline_parallel_size:
                num_warmup_microbatches = self.num_microbatches
                self.__str__all_warmup_microbatches = True
            else:
                num_warmup_microbatches = (self.pipeline_parallel_size - self.pipeline_parallel_rank - 1) * 2
                num_warmup_microbatches += (self.num_model_chunks - 1) * self.pipeline_parallel_size
                num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        self.all_warmup_microbatches = all_warmup_microbatches
        self.num_warmup_microbatches = num_warmup_microbatches
        self.num_microbatches_remaining = self.num_microbatches - self.num_warmup_microbatches

    def __call__(self):
        """Run 1F1B schedule with communication between pipeline stages.

        Returns:
            List of reduced loss torch.Tensor's if the last stage, empty dict otherwise.
        """
        self._warmup_forward()
        self.run_1f1b_steady_state()
        self.run_cooldown_backward()

        return self.losses_reduced

    def run_warmup_forward(self):
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        self.input_tensors[0].append(p2p_communication.recv_forward(timers=self.timers))
        for k in range(self.num_warmup_microbatches):
            output_tensor = self.forward_step_helper(k)

            # Determine if tensor should be received from previous stage.
            next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False
            if k == self.num_microbatches - 1:
                recv_prev = False

            # Don't send tensor downstream if on last stage.
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if k == (self.num_warmup_microbatches - 1) and not self.forward_only and not self.all_warmup_microbatches:
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    timers=self.timers,
                )
                self.output_tensor_grads[self.num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, timers=self.timers
                )
            self.input_tensors[next_forward_model_chunk_id].append(input_tensor)

    def run_1f1b_steady_state(self):
        # Run 1F1B in steady state.
        for k in range(self.num_microbatches_remaining):
            # Forward pass.
            forward_k = k + self.num_warmup_microbatches
            output_tensor = self.forward_step_helper(forward_k)

            # Backward pass.
            backward_k = k
            input_tensor_grad = self.backward_step_helper(backward_k)

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
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (self.pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (self.num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (self.pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (self.num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                timers=self.timers,
            )

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                self.input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                self.output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    def run_coolwodn_backward(self):
        # Run cooldown backward passes (flush out pipeline).
        if not self.forward_only:
            if self.all_warmup_microbatches:
                self.output_tensor_grads[self.num_model_chunks - 1].append(
                    p2p_communication.recv_backward(timers=self.timers)
                )
            for k in range(self.num_microbatches_remaining, self.num_microbatches):
                input_tensor_grad = self.backward_step_helper(k)
                next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_model_chunk_id == (self.num_model_chunks - 1):
                        recv_next = False
                if k == self.num_microbatches - 1:
                    recv_next = False
                self.output_tensor_grads[next_backward_model_chunk_id].append(
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad, recv_next=recv_next, timers=self.timers
                    )
                )

    def forward_step_helper(self, microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, self.num_model_chunks, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # forward step
        if (
                parallel_state.is_pipeline_first_stage() and
                len(self.input_tensors[model_chunk_id]) == len(self.output_tensors[model_chunk_id])
        ):
            self.input_tensors[model_chunk_id].append(None)
        input_tensor = self.input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(
            self.forward_step_func,
            self.data_iterator[model_chunk_id],
            self.model[model_chunk_id],
            input_tensor,
            self.losses_reduced,
        )
        self.output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if self.forward_only:
            self.input_tensors[model_chunk_id].pop()
            self.output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(self, microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if (
                parallel_state.is_pipeline_last_stage() and
                len(self.output_tensor_grads[model_chunk_id]) == 0
        ):
            self.output_tensor_grads[model_chunk_id].append(None)
        input_tensor = self.input_tensors[model_chunk_id].pop(0)
        output_tensor = self.output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = self.output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            self.optimizer, input_tensor, output_tensor, output_tensor_grad
        )
        return input_tensor_grad


def _forward_backward_pipelining_with_interleaving(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        timers,
        forward_only,
):
    return PipeliningWithInterleaving(forward_step_func, data_iterator, model, optimizer, timers, forward_only)()


def forward_backward_pipelining_with_interleaving(
    forward_step_func, data_iterator, model, optimizer, timers, forward_only
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = num_microbatches
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
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, num_model_chunks, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # forward step
        if (
                parallel_state.is_pipeline_first_stage() and
                len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id])
        ):
            input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            input_tensor,
            losses_reduced,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            optimizer, input_tensor, output_tensor, output_tensor_grad
        )

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(timers=timers))
    for k in range(num_warmup_microbatches):
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if k == (num_warmup_microbatches - 1) and not forward_only and not all_warmup_microbatches:
            input_tensor_grad = None
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                timers=timers,
            )
            output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
        else:
            input_tensor = p2p_communication.send_forward_recv_forward(
                output_tensor, recv_prev=recv_prev, timers=timers
            )
        input_tensors[next_forward_model_chunk_id].append(input_tensor)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        output_tensor = forward_step_helper(forward_k)

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
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

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
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        # Communicate tensors.
        (
            input_tensor,
            output_tensor_grad,
        ) = p2p_communication.send_forward_backward_recv_forward_backward(
            output_tensor,
            input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            timers=timers,
        )

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(timers=timers)
            )
        for k in range(num_microbatches_remaining, num_microbatches):
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
                    input_tensor_grad, recv_next=recv_next, timers=timers
                )
            )

    return losses_reduced


def forward_backward_pipelining_without_interleaving(
    forward_step_func, data_iterator, model, optimizer, timers, forward_only
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size() - parallel_state.get_pipeline_model_parallel_rank() - 1
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
        input_tensor = p2p_communication.recv_forward(timers=timers)
        output_tensor = forward_step(
            forward_step_func, data_iterator, model, input_tensor, losses_reduced
        )
        p2p_communication.send_forward(output_tensor, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward(timers=timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        output_tensor = forward_step(
            forward_step_func, data_iterator, model, input_tensor, losses_reduced
        )
        if forward_only:
            p2p_communication.send_forward(output_tensor, timers=timers)

            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(timers=timers)

        else:
            output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, timers=timers
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad
            )

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, timers=timers)
            else:
                input_tensor = p2p_communication.send_backward_recv_forward(
                    input_tensor_grad, timers=timers
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communication.recv_backward(timers=timers)

            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad
            )

            p2p_communication.send_backward(input_tensor_grad, timers=timers)

    return losses_reduced
