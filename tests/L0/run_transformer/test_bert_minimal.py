import torch
import unittest
from apex.transformer.testing import global_vars
from apex.transformer.testing.standalone_bert import bert_model_provider
from apex.transformer.pipeline_parallel.schedules.common import (
    _get_params_for_weight_decay_optimization, build_model
)
from apex.transformer.pipeline_parallel.schedules import get_forward_backward_func
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group, unwrap_model, setup_microbatch_calculator
)
from apex.transformer.log_util import set_logging_level
from apex.transformer import tensor_parallel, parallel_state
from apex.transformer.enums import ModelType
from apex.transformer._ucc_util import HAS_UCC
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase, NcclDistributedTestBase
import logging

from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)


logging.getLogger("apex").setLevel(logging.WARNING)


set_logging_level("WARNING")


class BertTestBase:

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
        temps = []
        for i in range(batch_size):
            if self.inds is None or self.data_idx >= len(self.inds):
                # hack as use of RNG will fall out of sync due to pipelines being different
                torch.manual_seed(self.MANUAL_SEED)
                self.inds = torch.randperm(
                    self.effective_length, device="cuda")
                self.masks = (
                    torch.rand(
                        len(self.inds) // batch_size + 1, batch_size, sequence_len, device="cuda"
                    )
                    >= self.MASK_PROB
                ).long()
                self.MANUAL_SEED += 1
                self.data_idx = 0
                if self.rank == 0:
                    print("new epoch", len(self.inds))
                    print("my start", self.inds[0:5])
                    print("masks_checksum:", torch.sum(self.masks))
            if self.EASY_MODE:
                data_idx_ = self.data_idx % self.EASY_MODE_SIZ
            else:
                data_idx_ = self.data_idx
            offset = self.inds[data_idx_]  # * SEQUENCE_LEN
            self.data_idx += 1

            curr = self.fancy_data[offset: offset +
                                   sequence_len].clone().detach()
            temps.append(curr)
        temp = torch.stack(temps, dim=0).cuda()
        mask = self.masks[self.data_idx // batch_size]
        mask_not = torch.logical_not(mask).long()
        data = mask * temp + mask_not * 124
        label = temp
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            data_dict = {"text": data, "label": label, "mask_not": mask_not}
        else:
            data_dict = None
        keys = ["text", "label", "mask_not"]
        broadcasted_data = tensor_parallel.broadcast_data(
            keys, data_dict, torch.long)
        return (
            broadcasted_data["text"].long(),
            broadcasted_data["label"].long(),
            broadcasted_data["mask_not"],
        )

    def _fwd_step_func(self, batch, model):
        data, label, loss_mask = batch
        y = model(data, torch.ones_like(data), lm_labels=label)

        def loss_func(output_tensor):
            output_tensor, _ = output_tensor
            lm_loss_ = output_tensor.float()
            lm_loss = torch.sum(lm_loss_.view(-1) *
                                loss_mask.reshape(-1)) / loss_mask.sum()
            averaged_loss = average_losses_across_data_parallel_group([
                                                                      lm_loss])
            if self.data_idx >= 1536:
                # NOTE (patwang): Loss cutoff might be excessively high but roughly one in five
                # unlucky random seeds do cause loss to spike to just under 8.0
                self.assertLess(averaged_loss, 8.0)
            return lm_loss, {"avg": averaged_loss}

        return y, loss_func

    def _train(
        self, model, optim, virtual_pipeline_model_parallel_size, pipeline_model_parallel_size, async_comm
    ):
        args = global_vars.get_args()
        sequence_len = args.seq_length
        micro_batch_size = args.micro_batch_size
        hidden_size = args.hidden_size
        global_batch_size = args.global_batch_size
        forward_backward_func = get_forward_backward_func(
            virtual_pipeline_model_parallel_size, pipeline_model_parallel_size
        )
        tensor_shape = (sequence_len, micro_batch_size, hidden_size)
        for _ in range(16):
            batch = self._generate_fancy_data_labels(
                sequence_len, global_batch_size)
            optim.zero_grad()
            forward_backward_func(
                self._fwd_step_func,
                batch,
                model,
                forward_only=False,
                tensor_shape=tensor_shape,
                async_comm=async_comm,
                sequence_parallel_enabled=args.sequence_parallel,
            )
            # All-reduce layernorm parameters across model parallel nodes
            # when sequence parallelism is used
            if parallel_state.get_tensor_model_parallel_world_size() > 1 and args.sequence_parallel:
                for model_module in model:
                    unwrapped_model = unwrap_model(model_module)
                    for param in unwrapped_model.parameters():
                        if getattr(param, 'sequence_parallel_enabled', False):
                            grad = param.grad
                            torch.distributed.all_reduce(
                                grad, group=parallel_state.get_tensor_model_parallel_group())

            optim.step()

    @unittest.skipUnless(torch.cuda.device_count() > 2, "requires at least 3 gpus")
    def test_bert_without_interleaving(self):
        self._test_bert(virtual_pipeline_model_parallel_size=None)

    @unittest.skipUnless(torch.cuda.device_count() > 2, "requires at least 3 gpus")
    def test_bert_with_interleaving(self):
        if self.DISTRIBUTED_BACKEND == 'ucc':
            self.skipTest('skip interleaving with ucc')
        self._test_bert(virtual_pipeline_model_parallel_size=2)

    def _test_bert(self, virtual_pipeline_model_parallel_size):

        self.MANUAL_SEED = 42
        self.inds = None
        self.masks = None
        self.data_idx = 0
        self.MASK_PROB = 0.1
        self.EASY_MODE = False
        self.EASY_MODE_SIZ = 32

        num_devices = torch.cuda.device_count()
        tensor_model_parallel_size = 2 if num_devices % 2 == 0 and num_devices > 4 else 1
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
            "bert_binary_head": False,
            "world_size": self.world_size,
            "rank": self.rank,
        }

        global_vars.set_global_variables(override_args=override_args, ignore_unknown_args=True)
        args = global_vars.get_args()

        self.fancy_data = self._download_fancy_data()
        self.effective_length = self.fancy_data.size(0) // args.seq_length
        self.effective_length = self.fancy_data.size(0) - args.seq_length

        if self.rank == 0:
            print(
                f'testing backend: {self.DISTRIBUTED_BACKEND} with virtual_pipeline_model_parallel_size: {virtual_pipeline_model_parallel_size}')
        async_comm = not args.sequence_parallel and virtual_pipeline_model_parallel_size is None
        self.data_idx = 0
        args.padded_vocab_size = 128  # needed in standalone gpt
        args.model_type = ModelType.encoder_or_decoder
        setup_microbatch_calculator(
            args.rank,
            args.rampup_batch_size,
            args.global_batch_size,
            args.micro_batch_size,
            args.data_parallel_size,
        )
        parallel_state.initialize_model_parallel(
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            default_backend="nccl",
            p2p_backend=self.DISTRIBUTED_BACKEND,
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
        self._train(
            model,
            optim,
            virtual_pipeline_model_parallel_size,
            args.pipeline_model_parallel_size,
            async_comm,
        )
        torch.cuda.synchronize()


class NcclBertTest(BertTestBase, NcclDistributedTestBase):
    pass


@unittest.skipUnless(HAS_UCC, "requires pytorch to be built with native ucc")
class UccBertTest(BertTestBase, UccDistributedTestBase):
    pass


if __name__ == "__main__":
    common_utils.run_tests()
