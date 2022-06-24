from functools import partial
from typing import List
import time

import torch
try:
    import torch_ucc
except ImportError:
    HAS_TORCH_UCC = False
else:
    HAS_TORCH_UCC = True
    print("Use UCC as backend of Pipeline Parallel ProcessGroups")

from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.tensor_parallel import model_parallel_cuda_manual_seed
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import unwrap_model
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group,
)
from apex.transformer.pipeline_parallel.utils import get_ltor_masks_and_position_ids
from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer.pipeline_parallel.schedules.common import (
    _get_params_for_weight_decay_optimization,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)
from apex.transformer.testing.standalone_gpt import gpt_model_provider
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE
from apex.transformer.testing.commons import initialize_distributed

MANUAL_SEED = 42
inds = None
data_idx = 0
N_VOCAB = 128


def download_fancy_data():
    # import requests
    # response = requests.get('https://internet.com/book.txt')
    # text = ' '.join(response.text.split())
    text = """
  An original sentence not subject to any license restrictions, copyright, or royalty payments. Nothing to see here. Commercial or non-commercial use. Research or non-research purposes. The quick brown fox jumps over the lazy dog. Lorem ipsum.
  """
    text = text * 1024
    encoded = text.encode("ascii", "replace")
    ints = [int(encoded[i]) for i in range(len(encoded))]
    return torch.tensor(ints)


# build a batch given sequence_len and batch size
def generate_fancy_data_labels(sequence_len, batch_size):
    global data_idx
    global inds
    global MANUAL_SEED
    temps = list()
    for i in range(batch_size):
        if inds is None or data_idx >= len(inds):
            # hack as use of RNG will fall out of sync due to pipelines being different
            model_parallel_cuda_manual_seed(MANUAL_SEED)
            inds = torch.randperm(effective_length, device="cuda")
            MANUAL_SEED += 1
            data_idx = 0
        data_idx_ = data_idx
        offset = inds[data_idx_]
        data_idx += 1
        curr = fancy_data[offset : offset + sequence_len + 1].clone().detach()
        temps.append(curr)
    temp = torch.stack(temps, dim=0).cuda()
    return temp


easy_data = None


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

    return loss, {"lm loss": averaged_loss[0]}


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/pretrain_gpt.py#L86
def fwd_step_func(batch, model):
    """Forward step."""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(batch)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train(model, optim, pipeline_model_parallel_size, async_comm):
    sequence_len = global_vars.get_args().seq_length
    micro_batch_size = global_vars.get_args().micro_batch_size
    hidden_size = global_vars.get_args().hidden_size
    fwd_bwd_func = forward_backward_pipelining_without_interleaving

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    runtime = 0
    # training loop
    for i in range(3):
        since = time.time()
        if torch.distributed.get_rank() == 0:
            print("begin iter", i)
        batch = [
            generate_fancy_data_labels(args.seq_length, args.global_batch_size)
            for _ in range(pipeline_model_parallel_size)
        ]
        if torch.distributed.get_rank() == 0:
            print("finished making batch...")
        optim.zero_grad()
        fwd_bwd_func(
            fwd_step_func,
            batch,
            model,
            forward_only=False,
            tensor_shape=tensor_shape,
            async_comm=async_comm,
            sequence_parallel_enabled=args.sequence_parallel,
        )
        if torch.distributed.get_rank() == 0:
            print("finished forward step")
        # All-reduce layernorm parameters across model parallel nodes
        # when sequence parallelism is used
        if parallel_state.get_tensor_model_parallel_world_size() > 1 and global_vars.get_args().sequence_parallel:
            for model_module in model:
                unwrapped_model = unwrap_model(model_module)
                for param in unwrapped_model.parameters():
                    if getattr(param, 'sequence_parallel_enabled', False):
                        grad = param.grad
                        torch.distributed.all_reduce(grad, group=parallel_state.get_tensor_model_parallel_group())
        optim.step()
        if torch.distributed.get_rank() == 0:
            print("finished iter", i)
        runtime += time.time() - since
    return runtime / 3.0


if __name__ == "__main__":
    init = True
    global_vars.set_global_variables()
    for async_comm in (False,) if global_vars.get_args().sequence_parallel else (False, True):
        global fancy_data
        global effective_length

        if init:
            init = False

            fancy_data = download_fancy_data()
            args = global_vars.get_args()
            args.model_type = ModelType.encoder_or_decoder
            effective_length = fancy_data.size(0) // args.seq_length
            effective_length = fancy_data.size(0) - args.seq_length

            initialize_distributed("nccl")
            world_size = torch.distributed.get_world_size()

            failure = None
            args.padded_vocab_size = 128
            batch_size = args.global_batch_size
            micro_batch_size = args.micro_batch_size
            setup_microbatch_calculator(
                args.rank,
                args.rampup_batch_size,
                args.global_batch_size,
                args.micro_batch_size,
                args.data_parallel_size,  # args.data_parallel_size,
            )
            world_size = torch.distributed.get_world_size()

        print(args.tensor_model_parallel_size, "MODEL PARALLEL SIZE")

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=args.tensor_model_parallel_size,
            pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
            default_backend="nccl",
            p2p_backend="ucc" if HAS_TORCH_UCC else "nccl",
        )

        pipeline_model_parallel_size = (
            parallel_state.get_pipeline_model_parallel_world_size()
        )
        model_parallel_cuda_manual_seed(0)
        model = build_model(
            gpt_model_provider,
            wrap_with_ddp=parallel_state.get_data_parallel_world_size() > 1,
            virtual_pipeline_model_parallel_size=None,
            cpu_offload=args.cpu_offload,
        )
        assert isinstance(model, list), model
        _param_groups = _get_params_for_weight_decay_optimization(model)
        optim = torch.optim.Adam(_param_groups)
        runtime = train(model, optim, args.pipeline_model_parallel_size, async_comm)

        parallel_state.destroy_model_parallel()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)
        print("Average Iteration Time:", runtime)
