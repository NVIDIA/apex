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

"""Utilities for pipeline model parallel."""
from typing import Optional, List, Union, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel

from apex.multi_tensor_apply import multi_tensor_applier
from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.microbatches import build_num_microbatches_calculator
from apex.transformer.pipeline_parallel._timers import _Timers
if multi_tensor_applier.available:
    import amp_C


_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_AUTORESUME = None
_GLOBAL_TIMERS = None


Shape = Union[List[int], torch.Size]


def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


def setup_microbatch_calculator(
        rank: int,
        rampup_batch_size: Optional[List[int]],
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
) -> None:
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, 'num microbatches calculator')

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size)


def _reconfigure_microbatch_calculator(
        rank: int,
        rampup_batch_size: Optional[List[int]],
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
) -> None:
    if torch.distributed.get_rank() == 0:
        import warnings
        warnings.warn("This function is only for unittest")
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size)


def get_micro_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_size


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check)


# note (mkozuki): Comment out in favor of `get_kth_microbatch`
def _split_batch_into_microbatch(
        batch: List[torch.Tensor],
        *,
        _micro_batch_size: Optional[int] = None,
        _global_batch_size: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    micro_batch_size = _micro_batch_size
    global_batch_size = _global_batch_size
    if micro_batch_size is None:
        micro_batch_size = get_micro_batch_size()
    if global_batch_size is None:
        global_batch_size = get_current_global_batch_size()
    for i in range(0, global_batch_size, micro_batch_size):
        yield [x[i * micro_batch_size:(i + 1) * micro_batch_size] for x in batch]


# TODO(mkozuki): Support non-tensor local minibatches?
def get_kth_microbatch(batch: Optional[List[torch.Tensor]], k: int) -> List[torch.Tensor]:
    """Create a list of microbatches from a list of local minibatches.

    This function creates a list of `k`th microbatches from a list of local minibatches.
    `a local minibatch` consists of `global_batch_size / data_parallel_size` samples.
    """
    if batch is None or not isinstance(batch, (List, Tuple)):
        return batch
    micro_batch_size = get_micro_batch_size()
    start = k * micro_batch_size
    end = start + micro_batch_size
    microbatch = list()
    for x in batch:
        size = x.size(0)
        assert size > start and size >= end
        microbatch.append(x[start:end])
    assert len(microbatch) > 0
    return microbatch


def get_autoresume():
    return _GLOBAL_AUTORESUME


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = _Timers()


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def print_rank_0(message: str) -> None:
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def param_is_not_shared(param: torch.nn.Parameter) -> bool:
    return getattr(param, "shared", False)


def unwrap_model(model, module_instances=(DistributedDataParallel,)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def get_model_type(
        model: torch.nn.Module,
) -> ModelType:
    """Get `model_type` of `model`.

    If ``model`` doesn't have ``model_type`` attribute, return ``ModelType.encoder_or_decoder``.

    Args:
        model
    """
    return getattr(unwrap_model(model), "model_type", ModelType.encoder_or_decoder)


def calc_params_l2_norm(model: torch.nn.Module, bf16: bool):
    """Calculate l2 norm of parameters """
    # args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = parallel_state.param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm, dummy_overflow_buf, [params_data], False  # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        norm_2, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_model_parallel_group()
    )
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=parallel_state.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(
        group=parallel_state.get_data_parallel_group()
    )

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if parallel_state.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = "iteration, rank, index, tensor-model-parallel, min, max, norm\n"
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group["params"]:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += "{:7d}, {:4d}, {:4d}, {:2d}, ".format(
                iteration, rank, index, int(param.tensor_model_parallel)
            )
            string += "{:.6E}, {:.6E}, {:.6E}\n".format(min_, max_, norm)
    print(string, flush=True)


# NOTE (mkozuki): APEX doesn't have anything equivalent for
# `_GLOBAL_ADLR_AUTORESUME` like Megatron-LM.
# def check_adlr_autoresume_termination(iteration, model, optimizer, lr_scheduler, save: bool):
#     """Check for autoresume signal and exit if it is received."""
#     from apex.ppu.checkpointing import save_checkpoint
#
#     autoresume = get_adlr_autoresume()
#     # Add barrier to ensure consistency.
#     torch.distributed.barrier()
#     if autoresume.termination_requested():
#         if save:
#             save_checkpoint(iteration, model, optimizer, lr_scheduler)
#         print_rank_0(">>> autoresume termination request found!")
#         if torch.distributed.get_rank() == 0:
#             autoresume.request_resume()
#         print_rank_0(">>> training terminated. Returning")
#         sys.exit(0)


def get_ltor_masks_and_position_ids(
    data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
