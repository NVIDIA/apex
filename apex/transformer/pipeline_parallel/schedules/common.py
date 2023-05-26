from typing import Any, Callable, Dict, List, Tuple, Union, Optional, Sequence

import torch
from torch.autograd.variable import Variable

from apex.normalization.fused_layer_norm import FusedLayerNorm
from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.pipeline_parallel.p2p_communication import FutureTensor
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import unwrap_model
from apex.transformer.pipeline_parallel.utils import get_model_type
from apex.transformer.tensor_parallel.layers import (
    set_defaults_if_not_set_tensor_model_parallel_attributes,
)
from apex.transformer.log_util import get_transformer_logger


_logger = get_transformer_logger(__name__)


Batch = Union[torch.Tensor, FutureTensor, List[Union[torch.Tensor, FutureTensor]], Tuple[Union[torch.Tensor, FutureTensor], ...]]
LossFunc = Callable[[torch.Tensor], torch.Tensor]
FwdStepFunc = Callable[
    [Optional[Batch], torch.nn.Module], Tuple[torch.Tensor, LossFunc]
]


def build_model(
    model_provider_func: Callable[[Any, Dict[str, Any]], torch.nn.Module],
    wrap_with_ddp: bool = True,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    model_type: ModelType = ModelType.encoder_or_decoder,
    *args: Any,
    **kwargs: Any,
) -> List[torch.nn.Module]:
    """Build the model satisfying pipeline model parallel requirements.

    This function sets `pre_process` and `post_process` to `**kwargs` and pass `*args` and `**kwargs` to
    `model_provider_func`.

    Args:
        model_provider_func: A function which takes `*args` and `**kwargs` and returns a `nn.Module`.
        wrap_with_ddp: If :obj:`True`, wrap the instantiated model
            with `torch.nn.parallel.distributed.DistributedDataParallel`, a.k.a. `DDP`.
        virtual_pipeline_model_parallel_size: Specify when using interleaving scheduling pipeline model parallel.
        model_type:
        *args: arguments for model provider func
        **kwargs: Keyword arguments for model provider func

    Returns:
        a list of `nn.Module`(s). If `virtual_pipeline_model_parallel_size` is not None,
        the list has multiple models, otherwise one.
    """
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and virtual_pipeline_model_parallel_size is not None
    ):
        model = []
        for i in range(virtual_pipeline_model_parallel_size):
            cur_args = args
            cur_kwargs = kwargs
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            cur_kwargs.update(
                {"pre_process": pre_process, "post_process": post_process,}
            )
            this_model = model_provider_func(*cur_args, **cur_kwargs)
            model.append(this_model)
    else:
        cur_args = args
        cur_kwargs = kwargs
        if model_type == ModelType.encoder_or_decoder:
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            cur_kwargs.update(
                {"pre_process": pre_process, "post_process": post_process,}
            )
            model = model_provider_func(*cur_args, **cur_kwargs)
        elif model_type == ModelType.encoder_and_decoder:
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            # `add_encoder` & `add_decoder` logic.
            add_encoder, add_decoder = True, True
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                split_rank = parallel_state.get_pipeline_model_parallel_split_rank()
                if split_rank is None:
                    raise RuntimeError(
                        "Split rank needs to be specified for model with both encoder and decoder."
                    )
                rank = parallel_state.get_pipeline_model_parallel_rank()
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = rank == (split_rank - 1) or rank == (world_size - 1)
                add_encoder = parallel_state.is_pipeline_stage_before_split()
                add_decoder = parallel_state.is_pipeline_stage_after_split()
            cur_kwargs.update(
                {
                    "pre_process": pre_process,
                    "post_process": post_process,
                    "add_encoder": add_encoder,
                    "add_decoder": add_decoder,
                }
            )
            model = model_provider_func(*cur_args, **cur_kwargs)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if (
        parallel_state.model_parallel_is_initialized()
        and parallel_state.get_data_parallel_rank() == 0
    ):
        msg = " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_pipeline_model_parallel_rank(),
            _calc_number_of_params(model),
        )
        print(msg, flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    if wrap_with_ddp:
        i = torch.cuda.current_device()
        model = [
            torch.nn.parallel.distributed.DistributedDataParallel(
                model_module,
                device_ids=[i],
                output_device=i,
                process_group=parallel_state.get_data_parallel_group(),
            )
            for model_module in model
        ]
    return model


def _calc_number_of_params(model: List[torch.nn.Module]) -> int:
    assert isinstance(model, list)
    return sum(
        [
            sum([p.nelement() for p in model_module.parameters()])
            for model_module in model
        ]
    )


def _get_params_for_weight_decay_optimization(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    *,
    no_weight_decay_modules=(FusedLayerNorm,),
) -> Dict[str, torch.nn.Parameter]:
    """Divide params into with-weight-decay and without-weight-decay groups.

    Layernorms and biases will have no weight decay but the rest will.
    """
    modules = listify_model(model)
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, no_weight_decay_modules):
                no_weight_decay_params["params"].extend(
                    [p for p in list(module_._parameters.values()) if p is not None]
                )
            else:
                weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n != "bias"
                    ]
                )
                no_weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n == "bias"
                    ]
                )

    return weight_decay_params, no_weight_decay_params


def free_output_tensor(
    output_tensors: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    deallocate_pipeline_outputs: bool = False,
) -> None:
    """Pseudo-free the output tensor's `.data` field.

    This method should be called right after the output tensor has been sent to the next
    pipeline stage. At this point, the output tensor is only useful for its `.grad_fn` field,
    and not its `.data`.
    """
    if not deallocate_pipeline_outputs:
        return
    if output_tensors is None:
        return
    if isinstance(output_tensors, torch.Tensor):
        output_tensors = [output_tensors]
    for output_tensor in output_tensors:
        output_tensor.data = torch.cuda.FloatTensor([0])


def custom_backward(output: torch.Tensor, grad_output: Optional[torch.Tensor]) -> None:
    """Directly call C++ autograd engine.

    To make the `free_output_tensor` optimization work, the C++ autograd engine must be called
    directly, bypassing PyTorch's `torch.autograd.backward`. PyTorch's `backward` checks that the
    output and grad have the same shape, while C++ `backward` does not.
    """
    assert (
        output.numel() == 1
    ), "output should be pseudo-freed in schedule, to optimize memory consumption"
    assert isinstance(output, torch.Tensor), "output == {}.".format(
        type(output).__name__
    )
    assert isinstance(
        grad_output, (torch.Tensor, type(None))
    ), "grad_outptu == {}.".format(type(grad_output).__name__)

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "Implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call C++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
    forward_step_func: FwdStepFunc,
    batch: Optional[Batch],
    model: torch.nn.Module,
    input_tensor: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    losses_reduced: List[torch.Tensor],
    dtype: torch.dtype,
    disable_autocast: bool = False,
    checkpoint_activations_micro_batch: Optional[bool] = None,
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from batch, otherwise passed-in input_tensor is used.

    Returns output tensor.

    Args:
        forward_step_func: Model specific function. This takes a minibatch and model as its arguments and
            returns the model's output and the loss function.
        batch: minibatch
        model: unwrappable model
        input_tensor:
        losses_reduced:
        dtype:
        disable_autocast:
        checkpoint_activations_micro_batch:

    Returns:
        output_tensor
    """
    # timers = get_timers()
    # timers("forward-compute").start()
    unwrapped_model = unwrap_model(model)
    model_type = get_model_type(unwrapped_model)
    # NOTE (mkozuki): The passed `model` is expected to implement `set_input_tensor`.
    # See https://github.com/NVIDIA/Megatron-LM/blob/5ac5571ba0265af4c491ee0af1508ca7589450c6/megatron/model/transformer.py#L679  # NOQA
    # for the details of `set_input_tensor`.
    unwrap_output_tensor = not isinstance(input_tensor, list)
    if unwrap_output_tensor:
        input_tensor = [input_tensor]

    input_tensor = [inp.get() if isinstance(inp, FutureTensor) else inp for inp in input_tensor]

    unwrapped_model.set_input_tensor(input_tensor)
    with torch.cuda.amp.autocast(
        enabled=not disable_autocast and dtype in (torch.half, torch.bfloat16),
        dtype=dtype,
    ):
        if checkpoint_activations_micro_batch is None:
            output_tensor, loss_func = forward_step_func(batch, model)
        else:
            output_tensor, loss_func = forward_step_func(batch, model, checkpoint_activations_micro_batch)
        if parallel_state.is_pipeline_last_stage():
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / get_num_microbatches()
            losses_reduced.append(loss_reduced)
    # timers("forward-compute").stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(
    input_tensor: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
    output_tensor_grad: Optional[torch.Tensor],
    model_type: ModelType,
    *,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    deallocate_pipeline_outputs: bool = False,
) -> Union[None, torch.Tensor, Sequence[torch.Tensor]]:
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage).

    Args:
        input_tensor:
        output_tensor:
        output_tensor_grad:
    Keyword Arguments:
        grad_scaler:
        deallocate_pipeline_outputs: Experimental.
    Returns:
        input_tensor_grad
    """

    # timers = get_timers()
    # timers("backward-compute").start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = not isinstance(input_tensor, list)
    if unwrap_input_tensor_grad:
        input_tensor = [input_tensor]

    input_tensor = [inp.get() if isinstance(inp, FutureTensor) else inp for inp in input_tensor]

    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]

    output_tensor = [out.get() if isinstance(out, FutureTensor) else out for out in output_tensor]

    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    output_tensor_grad = [ogr.get() if isinstance(ogr, FutureTensor) else ogr for ogr in output_tensor_grad]

    # Backward pass.
    if grad_scaler is not None and output_tensor_grad[0] is None:
        output_tensor[0] = grad_scaler.scale(output_tensor[0])
    if deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            input_tensor_grad.append(None if x is None else x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            # todo (mkozuki): Replace the inplace add with `+= output_tensor_grad[1]`?
            input_tensor_grad[-1].add_(output_tensor_grad[1])

    # timers("backward-compute").stop()
    return input_tensor_grad[0] if unwrap_input_tensor_grad else input_tensor_grad
