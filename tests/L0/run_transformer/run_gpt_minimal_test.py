import random
import torch
from typing import List
from apex.transformer import tensor_parallel
from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import vocab_parallel_cross_entropy
from apex.transformer.tensor_parallel import model_parallel_cuda_manual_seed

from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import average_losses_across_data_parallel_group
#from apex.transformer.pipeline_parallel.utils import update_num_microbatches
from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer.pipeline_parallel.schedules.common import _get_params_for_weight_decay_optimization
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import _forward_backward_pipelining_with_interleaving

from apex.transformer.testing.standalone_gpt import post_language_model_processing, gpt_model_provider 
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator

mode = None
MANUAL_SEED = 42
inds = None
masks = None
data_idx = 0
MASK_PROB = 0.1
EASY_MODE = False
N_VOCAB = 128
# download a public domain book as corpus
def download_fancy_data():
  import requests
  response = requests.get('https://www.gutenberg.org/files/1342/1342-0.txt')
  #response = requests.get('https://www.gutenberg.org/files/84/84-0.txt')
  text = ' '.join(response.text.split())
  encoded = text.encode('ascii', 'replace')
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
      inds = torch.randperm(effective_length, device='cuda')
      MANUAL_SEED += 1
      print("new epoch", len(inds))
      data_idx = 0
    data_idx_ = data_idx
    offset = inds[data_idx_] #* SEQUENCE_LEN
    data_idx += 1
    curr = fancy_data[offset:offset+sequence_len].clone().detach()
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

    return loss, {'lm loss': averaged_loss[0]}


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/pretrain_gpt.py#L86
# TODO (mkozuki): Currently I'm seeing no attribute `word_embeddings` which looks weird.
def fwd_step_func(batch, model):
    """Forward step."""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(batch)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train(model, optim, virtual_pipeline_model_parallel_size):
    sequence_len = global_vars.get_args().seq_length
    micro_batch_size = global_vars.get_args().micro_batch_size
    hidden_size = global_vars.get_args().hidden_size
    if virtual_pipeline_model_parallel_size is None:
        fwd_bwd_func = forward_backward_pipelining_without_interleaving
    else:
        fwd_bwd_func = _forward_backward_pipelining_with_interleaving
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    for _ in range(8):
      if virtual_pipeline_model_parallel_size is None:
          batch = [generate_fancy_data_labels(args.seq_length, args.global_batch_size)]
      else:
          batch = [generate_fancy_data_labels(args.seq_length, args.global_batch_size) for _ in range(virtual_pipeline_model_parallel_size)]
      optim.zero_grad()
      forward_backward_func(fwd_step_func, batch, model, forward_only=False, tensor_shape=tensor_shape)
      optim.step()

if __name__ == '__main__':
    global fancy_data
    global effective_length

    global_vars.set_global_variables()

    fancy_data = download_fancy_data()
    effective_length = fancy_data.size(0) // global_vars.get_args().seq_length
    effective_length = fancy_data.size(0) - global_vars.get_args().seq_length

    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    failure = None
    if True:
        args = global_vars.get_args()
        args.padded_vocab_size = 128
        batch_size = args.global_batch_size
        micro_batch_size = args.micro_batch_size
        setup_microbatch_calculator(
            args.rank,
            args.rampup_batch_size,
            args.global_batch_size,
            args.micro_batch_size,
            1,  # args.data_parallel_size,
        )
        virtual_pipeline_model_parallel_size = args.pipeline_model_parallel_size
        world_size = torch.distributed.get_world_size()
        pipeline_model_parallel_size = world_size
        parallel_state.initialize_model_parallel(
            1, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size)
        pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        tensor_parallel.random.model_parallel_cuda_manual_seed(0)
        model = build_model(
            gpt_model_provider,
            wrap_with_ddp=True,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        )
        assert isinstance(model, list), model
        assert len(model) == (1 if virtual_pipeline_model_parallel_size is None else virtual_pipeline_model_parallel_size), len(model)
        _param_groups = _get_params_for_weight_decay_optimization(model)
        optim = torch.optim.Adam(_param_groups)
        print(effective_length)
        print(fancy_data.size(0))
        train(model, optim, virtual_pipeline_model_parallel_size)
    # except Exception as e:
    #     failure = str(e)
    # finally:
    #     parallel_state.destroy_model_parallel()
    #     if failure is not None:
    #         torch.distributed.barrier()
    #         if torch.distributed.get_rank() == 0:
    #             print(f"Minimal GPT Pipeline Parallel Failed with {failure}")
    #     else:
    #         torch.distributed.barrier()
    #         if torch.distributed.get_rank() == 0:
    #             print(TEST_SUCCESS_MESSAGE)