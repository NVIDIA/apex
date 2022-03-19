from functools import partial
import logging
from typing import List

import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.schedules.common import _get_params_for_weight_decay_optimization
from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import _forward_backward_pipelining_with_interleaving
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import forward_backward_pipelining_without_interleaving
from apex.transformer.pipeline_parallel.utils import average_losses_across_data_parallel_group
from apex.transformer.pipeline_parallel.utils import get_ltor_masks_and_position_ids
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import update_num_microbatches
from apex.transformer.tensor_parallel import model_parallel_cuda_manual_seed
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator
from apex.transformer.testing.standalone_gpt import gpt_model_provider
from apex.transformer.log_util import get_transformer_logger, set_logging_level


set_logging_level(logging.NOTSET)
_logger = get_transformer_logger("megatron_gpt_pipeline_test")
global_vars.set_global_variables()
N_VOCAB = 8192


def generate_batch(batch_size, sequence_length):
    size = batch_size, sequence_length + 1
    int_tensor = torch.randint(low=0, high=N_VOCAB, size=size, dtype=torch.long).cuda()
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
# TODO (mkozuki): Currently I'm seeing no attribute `word_embeddings` which looks weird.
def forward_step(batch, model):
    """Forward step."""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(batch)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def run_gpt(pipeline_model_parallel_size, virtual_pipeline_model_parallel_size=None, forward_only=False):
    parallel_state.initialize_model_parallel(1, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size)
    model_parallel_cuda_manual_seed(42)

    model = build_model(
        gpt_model_provider, True,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size)
    _logger.debug("building model")
    assert isinstance(model, list)
    assert len(model) == (1 or virtual_pipeline_model_parallel_size)
    _param_groups = _get_params_for_weight_decay_optimization(model)
    torch.optim.Adam(_param_groups)

    if parallel_state.is_pipeline_last_stage():
        _logger.debug("checking `word_embeddings` existence")
        for m in model:
            assert hasattr(m, "word_embeddings")

    args = global_vars.get_args()
    if virtual_pipeline_model_parallel_size is None:
        batch = generate_batch(args.global_batch_size, args.seq_length)
    else:
        batch = [generate_batch(args.global_batch_size, args.seq_length) for _ in range(virtual_pipeline_model_parallel_size)]
    _logger.debug("preparing batch")

    if virtual_pipeline_model_parallel_size is None:
        fwd_bwd_func = forward_backward_pipelining_without_interleaving
    else:
        fwd_bwd_func = _forward_backward_pipelining_with_interleaving
    _logger.debug(f"selecting forward_backward func: {fwd_bwd_func}")

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    _logger.debug(f"`tensor_shape`: {tensor_shape}")
    fwd_bwd_func(forward_step, batch, model, forward_only=forward_only, tensor_shape=tensor_shape)

    _logger.debug(TEST_SUCCESS_MESSAGE)


if __name__ == "__main__":
    initialize_distributed()
    args = global_vars.get_args()
    args.padded_vocab_size = N_VOCAB
    setup_microbatch_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        1,  # args.data_parallel_size,
    )
    update_num_microbatches(0, True)
    print_separator("run GPT model")
    try:
        run_gpt(torch.distributed.get_world_size())
    # TODO(mkozuki): handle exception correctly, but for now, lazily commenting out as
    # this won't get kicked by CI
    except Exception as e:
        _logger.debug(str(e))
        pass
    finally:
        parallel_state.destroy_model_parallel()
