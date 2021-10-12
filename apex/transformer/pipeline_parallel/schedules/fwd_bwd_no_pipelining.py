from typing import List, Union, Optional

import torch

from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.schedules.common import Batch, FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import placeholder_handler
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import backward_step


# note (mkozuki): tensor_shape is here for consistency with the other two pipeline parallel functions.
def forward_backward_no_pipelining(
        forward_step_func: FwdStepFunc,
        batch: Batch,
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        *,
        forward_only: bool,
        tensor_shape: Optional[Union[List[int], torch.Size]] = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Args:
        forward_step_func: A function which takes a minibatch and model as its arguments and
            returns model's forward output and the loss function.
            The loss function is supposed to take one `torch.Tensor` and
            return a `torch.Tensor` of loss and a dictionary of `str` and `torch.Tensor`.
        batch: A minibatch, i.e., a list of `torch.Tensor`'s.
        model: A `torch.nn.Module` or a list of `torch.nn.Module`.
        forward_only:
        tensor_shape: Shape of tensor.

    Returns:
        a list of loss `torch.Tensor`s if the last stage, empty list otherwise.
    """
    model = listify_model(model)
    if len(model) != 1:
        msg = f"`model` is expected be a `nn.Module`, but {type(model)}"
        raise RuntimeError(msg)
    model = model[0]

    context_handler = placeholder_handler
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        context_handler = model.no_sync

    losses_reduced = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(
                forward_step_func, batch, model, input_tensor, losses_reduced)
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(
        forward_step_func, batch, model, input_tensor, losses_reduced
    )
    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad)

    return losses_reduced
