# NOTE (mkozuki): For simplicity, tentatively `timers` related operations are commented out.
from contextlib import contextmanager
from typing import Callable, List, Tuple, Union, Optional


import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import unwrap_model


Batch = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
LossFunc = Callable[[torch.Tensor], torch.Tensor]
FwdStepFunc = Callable[[Batch, torch.nn.Module], Tuple[torch.Tensor, LossFunc]]


@contextmanager
def placeholder_handler():
    try:
        yield
    finally:
        pass


def forward_step(
        forward_step_func: FwdStepFunc,
        batch: Batch,
        model: torch.nn.Module,
        input_tensor: Optional[torch.Tensor],
        losses_reduced: List[torch.Tensor],
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor.

    Args:
        forward_step_func: Model specific function. This takes a minibatch and model as its arguments and
            returns the model's output and the loss function.
        batch: minibatch
        model: unwrappable model
        input_tensor:
        losses_reduced:
    Returns:
        output_tensor
    """
    # timers = get_timers()
    # timers("forward-compute").start()
    unwrapped_model = unwrap_model(model)
    # NOTE (mkozuki): The passed `model` is expected to implement `set_input_tensor`.
    # See https://github.com/NVIDIA/Megatron-LM/blob/5ac5571ba0265af4c491ee0af1508ca7589450c6/megatron/model/transformer.py#L679  # NOQA
    # for the details of `set_input_tensor`.
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor, loss_func = forward_step_func(batch, model)
    if parallel_state.is_pipeline_last_stage():
        output_tensor = loss_func(output_tensor)
        loss, loss_reduced = output_tensor
        output_tensor = loss / get_num_microbatches()
        losses_reduced.append(loss_reduced)
    # timers("forward-compute").stop()

    return output_tensor


def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage).

    Args:
        input_tensor:
        output_tensor:
        output_tensor_grad:
    Returns:
        input_tensor_grad
    """

    # timers = get_timers()
    # timers("backward-compute").start()
    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()
    # Backward pass.
    # if output_tensor_grad is None:
    #     output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
    # timers("backward-compute").stop()

    return input_tensor_grad
