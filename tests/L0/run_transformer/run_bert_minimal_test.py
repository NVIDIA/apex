import random
import torch
try:
    import torch_ucc
except ImportError:
    HAS_TORCH_UCC = False
else:
    HAS_TORCH_UCC = True
    print("Use UCC as backend of Pipeline Parallel ProcessGroups")

from apex.transformer.enums import ModelType
from apex.transformer import tensor_parallel
from apex.transformer import parallel_state
from apex.transformer.log_util import set_logging_level
from apex.transformer.tensor_parallel import vocab_parallel_cross_entropy
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import unwrap_model
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group,
)
from apex.transformer.pipeline_parallel.schedules import get_forward_backward_func
from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer.pipeline_parallel.schedules.common import (
    _get_params_for_weight_decay_optimization,
)

from apex.transformer.testing.standalone_bert import bert_model_provider
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator

import warnings


class DebugWarning(Warning):
    pass


set_logging_level("WARNING")
mode = None
MANUAL_SEED = 42
inds = None
masks = None
data_idx = 0
MASK_PROB = 0.1
EASY_MODE = False
EASY_MODE_SIZ = 32
ONCE = False


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
    global masks
    global MANUAL_SEED
    temps = []
    for i in range(batch_size):
        if inds is None or data_idx >= len(inds):
            # hack as use of RNG will fall out of sync due to pipelines being different
            torch.manual_seed(MANUAL_SEED)
            inds = torch.randperm(effective_length, device="cuda")
            masks = (
                torch.rand(
                    len(inds) // batch_size + 1, batch_size, sequence_len, device="cuda"
                )
                >= MASK_PROB
            ).long()
            MANUAL_SEED += 1
            print("new epoch", len(inds))
            data_idx = 0
            print("my start", inds[0:5])
            print("masks_checksum:", torch.sum(masks))
        if EASY_MODE:
            data_idx_ = data_idx % EASY_MODE_SIZ
        else:
            data_idx_ = data_idx
        offset = inds[data_idx_]  # * SEQUENCE_LEN
        data_idx += 1

        curr = fancy_data[offset : offset + sequence_len].clone().detach()
        temps.append(curr)
    temp = torch.stack(temps, dim=0).cuda()
    mask = masks[data_idx // batch_size]
    mask_not = torch.logical_not(mask).long()
    data = mask * temp + mask_not * 124
    label = temp
    if parallel_state.get_tensor_model_parallel_rank() == 0:
        data_dict = {"text": data, "label": label, "mask_not": mask_not}
    else:
        data_dict = None
    keys = ["text", "label", "mask_not"]
    dtype = torch.int64
    broadcasted_data = tensor_parallel.broadcast_data(keys, data_dict, torch.long)
    return (
        broadcasted_data["text"].long(),
        broadcasted_data["label"].long(),
        broadcasted_data["mask_not"],
    )


easy_data = None


def fwd_step_func(batch, model):
    data, label, loss_mask = batch
    y = model(data, torch.ones_like(data), lm_labels=label)

    def loss_func(output_tensor):
        global ONCE
        output_tensor, _ = output_tensor
        lm_loss_ = output_tensor.float()
        lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([lm_loss])
        if data_idx >= 1536:
            assert averaged_loss < 4.8
            if not ONCE:
                print("LOSS OK")
                ONCE = True
        return lm_loss, {"avg": averaged_loss}

    return y, loss_func


def train(
    model, optim, virtual_pipeline_model_parallel_size, pipeline_model_parallel_size, async_comm
):
    sequence_len = global_vars.get_args().seq_length
    micro_batch_size = global_vars.get_args().micro_batch_size
    hidden_size = global_vars.get_args().hidden_size
    forward_backward_func = get_forward_backward_func(
        virtual_pipeline_model_parallel_size, pipeline_model_parallel_size
    )
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    for _ in range(16):
        batch = generate_fancy_data_labels(sequence_len, batch_size)
        optim.zero_grad()
        forward_backward_func(
            fwd_step_func,
            batch,
            model,
            forward_only=False,
            tensor_shape=tensor_shape,
            async_comm=async_comm,
            sequence_parallel_enabled=global_vars.get_args().sequence_parallel,
        )
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


if __name__ == "__main__":
    global fancy_data
    global effective_length

    global_vars.set_global_variables()

    fancy_data = download_fancy_data()
    effective_length = fancy_data.size(0) // global_vars.get_args().seq_length
    effective_length = fancy_data.size(0) - global_vars.get_args().seq_length

    initialize_distributed("nccl")
    world_size = torch.distributed.get_world_size()
    failure = None
    init = True
    try:
        virtual_pipeline_model_parallel_sizes = (None, 2,)
        if HAS_TORCH_UCC:
            # Deliberately skipping test with interleaved schedule for BERT model.
            # It deadlocks on hybrid UCC/NCCL backend.
            virtual_pipeline_model_parallel_sizes = (None,)
        for virtual_pipeline_model_parallel_size in virtual_pipeline_model_parallel_sizes:
            args = global_vars.get_args()
            async_comm = not args.sequence_parallel and virtual_pipeline_model_parallel_size is None
            data_idx = 0
            ONCE = False
            if init:
                init = False
                args = global_vars.get_args()
                args.padded_vocab_size = 128  # needed in standalone gpt
                args.model_type = ModelType.encoder_or_decoder
                batch_size = args.global_batch_size
                micro_batch_size = args.micro_batch_size
                setup_microbatch_calculator(
                    args.rank,
                    args.rampup_batch_size,
                    args.global_batch_size,
                    args.micro_batch_size,
                    args.data_parallel_size,
                )
            else:
               parallel_state.destroy_model_parallel()
            parallel_state.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                default_backend="nccl",
                p2p_backend="ucc" if HAS_TORCH_UCC else "nccl",
            )
            pipeline_model_parallel_size = (
                parallel_state.get_pipeline_model_parallel_world_size()
            )

            tensor_parallel.random.model_parallel_cuda_manual_seed(0)
            model = build_model(
                bert_model_provider,
                wrap_with_ddp=parallel_state.get_data_parallel_world_size() > 1,
                virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
                cpu_offload=args.cpu_offload,
            )
            assert isinstance(model, list)
            assert len(model) == (
                1
                if virtual_pipeline_model_parallel_size is None
                else virtual_pipeline_model_parallel_size
            )
            _param_groups = _get_params_for_weight_decay_optimization(model)
            optim = torch.optim.Adam(_param_groups)
            print(effective_length)
            print(fancy_data.size(0))
            train(
                model,
                optim,
                virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_size,
                async_comm,
            )
    except Exception as e:
        failure = str(e)
    finally:
        parallel_state.destroy_model_parallel()
    if failure is not None:
        warnings.warn(
            f"Minimal BERT Pipeline Parallel Failed with: {failure}", DebugWarning
        )
        print(f"Minimal BERT Pipeline Parallel Failed with: {failure}")
    torch.distributed.barrier()
    print(TEST_SUCCESS_MESSAGE)
