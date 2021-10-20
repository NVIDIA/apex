from contextlib import contextmanager
from typing import List, Union

import torch

from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import get_kth_microbatch
from apex.transformer.pipeline_parallel.schedules.common import Batch, FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import backward_step


@contextmanager
def placeholder_handler():
    try:
        yield
    finally:
        pass


# TODO (mkozuki): Confirm this will be used or not.
# TODO (mkozuki): Fix if necessary. Currently I'm seeing failure if `not forward_only` and
#   the last `backward_step` seems to fail. However, note the possibility of my test script is wrong.
def forward_backward_no_pipelining(
        forward_step_func: FwdStepFunc,
        batch: Batch,
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        *,
        forward_only: bool,
        **kwargs,
):
    """Run forward and backward passes with no pipeline parallelism (no inter-stage communication).

    This pipeline parallel scheduling handles the last microbatch differently to synchronize gradients.

    Args:
        forward_step_func: A function which takes a minibatch and model as its arguments and
            returns model's forward output and the loss function.
            The loss function is supposed to take one `torch.Tensor` and
            return a `torch.Tensor` of loss and a dictionary of `str` and `torch.Tensor`.
        batch: A List of torch.Tensors
        model: A `torch.nn.Module` or a list of `torch.nn.Module`.

    Keyword args:
        forward_only:
        **kwargs: Added to handle `tensor_shape` which has no effect on this function.

    Returns:
        a list of dictionaries of loss `torch.Tensor`s if the last stage, empty list otherwise.
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
    num_micro_batches = get_num_microbatches()
    with context_handler():
        for i in range(num_micro_batches - 1):
            cur_micro_batch = get_kth_microbatch(batch, i)
            output_tensor = forward_step(
                forward_step_func, cur_micro_batch, model, input_tensor, losses_reduced)
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(
        forward_step_func, get_kth_microbatch(batch, num_micro_batches - 1), model, input_tensor, losses_reduced
    )
    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad)

    return losses_reduced
