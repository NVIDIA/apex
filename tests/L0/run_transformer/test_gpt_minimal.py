from functools import partial
from typing import List
import time

import torch

import unittest

from apex.transformer._ucc_util import HAS_UCC
from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.tensor_parallel import model_parallel_cuda_manual_seed
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group, unwrap_model, setup_microbatch_calculator,
    get_ltor_masks_and_position_ids
)
from apex.transformer.pipeline_parallel.schedules.common import (
    _get_params_for_weight_decay_optimization, build_model
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)
from apex.transformer.testing.standalone_gpt import gpt_model_provider
from apex.transformer.testing import global_vars

from apex.transformer.testing.distributed_test_base import UccDistributedTestBase, NcclDistributedTestBase

from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests


class GptTestBase:

    def _download_fancy_data(self):
        text = """
    An original sentence not subject to any license restrictions, copyright, or royalty payments. Nothing to see here. Commercial or non-commercial use. Research or non-research purposes. The quick brown fox jumps over the lazy dog. Lorem ipsum.
    """
        text = text * 1024
        encoded = text.encode("ascii", "replace")
        ints = [int(encoded[i]) for i in range(len(encoded))]
        return torch.tensor(ints)

    # build a batch given sequence_len and batch size
    def _generate_fancy_data_labels(self, sequence_len, batch_size):
        temps = list()
        for i in range(batch_size):
            if self.inds is None or self.data_idx >= len(self.inds):
                # hack as use of RNG will fall out of sync due to pipelines being different
                model_parallel_cuda_manual_seed(self.MANUAL_SEED)
                self.inds = torch.randperm(effective_length, device="cuda")
                self.MANUAL_SEED += 1
                self.data_idx = 0
            data_idx_ = self.data_idx
            offset = self.inds[data_idx_]
            self.data_idx += 1
            curr = fancy_data[offset: offset +
                              sequence_len + 1].clone().detach()
            temps.append(curr)
        temp = torch.stack(temps, dim=0).cuda()
        return temp

    def _get_batch(self, int_tensors: List[torch.Tensor]):
        data = int_tensors[0]
        # Unpack.
        tokens_ = data.long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        # Get the masks and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.N_VOCAB,  # tokenizer.eod,
            False,  # args.reset_position_ids,
            False,  # args.reset_attention_mask,
            False,  # args.eod_mask_loss,
        )
        return tokens, labels, loss_mask, attention_mask, position_ids

    # Ref: https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/pretrain_gpt.py#L75
    def _loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {"lm loss": averaged_loss[0]}

    # Ref: https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/pretrain_gpt.py#L86
    def _fwd_step_func(self, batch, model):
        """Forward step."""
        tokens, labels, loss_mask, attention_mask, position_ids = self._get_batch(
            batch)
        output_tensor = model(tokens, position_ids,
                              attention_mask, labels=labels)
        return output_tensor, partial(self._loss_func, loss_mask)

    def _train(self, model, optim, pipeline_model_parallel_size, async_comm):
        args = global_vars.get_args()
        fwd_bwd_func = forward_backward_pipelining_without_interleaving

        tensor_shape = (args.seq_length, args.micro_batch_size,
                        args.hidden_size)
        runtime = 0
        # training loop
        for i in range(3):
            since = time.time()
            if torch.distributed.get_rank() == 0:
                print("begin iter", i)
            batch = [
                self._generate_fancy_data_labels(
                    args.seq_length, args.global_batch_size)
                for _ in range(pipeline_model_parallel_size)
            ]
            if torch.distributed.get_rank() == 0:
                print("finished making batch...")
            optim.zero_grad()
            fwd_bwd_func(
                self._fwd_step_func,
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
                            torch.distributed.all_reduce(
                                grad, group=parallel_state.get_tensor_model_parallel_group())
            optim.step()
            if torch.distributed.get_rank() == 0:
                print("finished iter", i)
            runtime += time.time() - since
        return runtime / 3.0

    @unittest.skipUnless(torch.cuda.device_count() > 2, "requires at least 3 gpus")
    def test_gpt(self):
        self.MANUAL_SEED = 42
        self.inds = None
        self.data_idx = 0
        self.N_VOCAB = 128
        init = True

        num_devices = torch.cuda.device_count()
        tensor_model_parallel_size = 2 if num_devices % 2 == 0 and num_devices >= 4 else 1
        pipeline_model_parallel_size = num_devices // tensor_model_parallel_size

        override_args = {
            "micro_batch_size": 2,
            "num_layers": 16,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "max_position_embeddings": 512,
            "seq_length": 512,
            "global_batch_size": 128,
            "pipeline_model_parallel_size": pipeline_model_parallel_size,
            "tensor_model_parallel_size": tensor_model_parallel_size,
            "world_size": self.world_size,
            "rank": self.rank,
        }

        global_vars.set_global_variables(override_args=override_args, ignore_unknown_args=True)
        args = global_vars.get_args()

        for async_comm in (False,) if args.sequence_parallel else (False, True):
            global fancy_data
            global effective_length

            if init:
                init = False

                fancy_data = self._download_fancy_data()
                args = global_vars.get_args()
                args.model_type = ModelType.encoder_or_decoder
                effective_length = fancy_data.size(0) // args.seq_length
                effective_length = fancy_data.size(0) - args.seq_length

                args.padded_vocab_size = 128
                setup_microbatch_calculator(
                    args.rank,
                    args.rampup_batch_size,
                    args.global_batch_size,
                    args.micro_batch_size,
                    args.data_parallel_size,
                )

            print(args.tensor_model_parallel_size, "MODEL PARALLEL SIZE")

            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size_=args.tensor_model_parallel_size,
                pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
                default_backend="nccl",
                p2p_backend=self.DISTRIBUTED_BACKEND,
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
            runtime = self._train(
                model, optim, args.pipeline_model_parallel_size, async_comm)

            parallel_state.destroy_model_parallel()
        torch.cuda.synchronize()


class NcclGptTest(GptTestBase, NcclDistributedTestBase):
    pass


@unittest.skipUnless(HAS_UCC, "requires pytorch to be built with native ucc")
class UccGptTest(GptTestBase, UccDistributedTestBase):
    pass


if __name__ == "__main__":
    common_utils.run_tests()
