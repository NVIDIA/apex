from functools import partial
from typing import List

import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import update_num_microbatches
from apex.transformer.pipeline_parallel.utils import average_losses_across_data_parallel_group
from apex.transformer.pipeline_parallel.utils import get_ltor_masks_and_position_ids
from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer.pipeline_parallel.schedules.common import _get_params_for_weight_decay_optimization
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import forward_backward_pipelining_with_interleaving
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import forward_backward_pipelining_without_interleaving

from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.standalone_gpt import gpt_model_provider


global_vars.set_global_variables()


fwd_bwd_functions = {
    "no_pipelining": forward_backward_no_pipelining,
    "no_interleaving": forward_backward_pipelining_without_interleaving,
    "interleaving": forward_backward_pipelining_with_interleaving,
}

N_VOCAB = 8192


def generate_batch(batch_size, sequence_length):
    size = batch_size, sequence_length
    int_tensor = torch.randint(low=0, high=N_VOCAB, size=size, dtype=torch.long)
    return int_tensor,


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/pretrain_gpt.py#L44
def get_batch(int_tensors: List[torch.Tensor]):
    data = int_tensors[0]
    # Unpack.
    tokens_ = data.long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        N_VOCAB,  # tokenizer.eod,
        False,  # args.reset_position_ids,
        False,  # args.reset_attention_mask,
        False,  # args.eod_mask_loss,
    )
    return tokens, labels, loss_mask, attention_mask, position_ids


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/pretrain_gpt.py#L75
def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/pretrain_gpt.py#L86
def forward_step(batch, model):
    """Forward step."""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(batch)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def run_gpt(pipeline_model_parallel_size, virtual_pipeline_model_parallel_size=None, forward_only=False):
    parallel_state.initialize_model_parallel(1, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size)

    model = build_model(
        gpt_model_provider, False,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size)
    _param_groups = _get_params_for_weight_decay_optimization(model)
    torch.optim.Adam(_param_groups)

    args = global_vars.get_args()
    if virtual_pipeline_model_parallel_size is None:
        batch = generate_batch(args.global_batch_size, args.seq_length)
    else:
        batch = [generate_batch(args.global_batch_size, args.seq_length) for _ in range(virtual_pipeline_model_parallel_size)]

    if virtual_pipeline_model_parallel_size is None:
        fwd_bwd_func = forward_backward_pipelining_without_interleaving
    else:
        fwd_bwd_func = forward_backward_pipelining_with_interleaving

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    fwd_bwd_func(forward_step, batch, model, forward_only=forward_only, tensor_shape=tensor_shape)


if __name__ == "__name__":
    initialize_distributed()
    args = global_vars.get_args()
    setup_microbatch_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        1,  # args.data_parallel_size,
    )
    update_num_microbatches(0, True)
    run_gpt(torch.distributed.get_world_size())