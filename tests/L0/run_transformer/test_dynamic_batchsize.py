from typing import Tuple, List

import torch
import unittest

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.schedules.common import (
    _get_params_for_weight_decay_optimization, build_model
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import (
    _forward_backward_pipelining_with_interleaving,
)
from apex.transformer.pipeline_parallel.utils import (
    setup_microbatch_calculator, _reconfigure_microbatch_calculator, update_num_microbatches
)
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import (
    print_separator, fwd_step_func, model_provider_func
)
from apex.transformer.log_util import get_transformer_logger
from apex.transformer._data import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase

from torch.testing._internal import common_utils

# note(mkozuki): To see warmup, steady, cooldown iterations, uncomment the line below
# set_logging_level("INFO")
_logger = get_transformer_logger("pipeline_parallel_test")
# note(mkozuki): To see if local batch size increases, uncomment the line below
# _logger.setLevel("INFO")


NUM_ITERATIONS = 20
NUM_SAMPLES = 16384 // 2
HIDDEN_SIZE = 16


def Dataset(num_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.randn(HIDDEN_SIZE // 2, HIDDEN_SIZE // 2),
        )
        for _ in range(num_samples)
    ]


# Run forward & backward with dynamic batch size.
def run_interleaved_with_dynamic_batch_size(
    pipeline_model_parallel_size: int, forward_only: bool, BatchSamplerCls,
) -> None:
    args = global_vars.get_args()
    _reconfigure_microbatch_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        1,  # args.data_parallel_size,
    )
    virtual_pipeline_model_parallel_size = 2
    # NOTE (mkozuki): `virtual_pipeline_model_parallel_size` is a requisite for the interleaving scheduling
    # In megatron, `args.virtual_pipeline_model_parallel_size` is computed in megatron/arguments.py and
    # used ubiquitously but this test uses custom model so it's safe to abuse.
    parallel_state.initialize_model_parallel(
        1, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size
    )
    pipeline_model_parallel_size = (
        parallel_state.get_pipeline_model_parallel_world_size()
    )

    print_separator(
        f"BatchSamplerCls: {BatchSamplerCls.__name__}, forward_only: {forward_only}"
    )

    model = build_model(
        model_provider_func,
        wrap_with_ddp=True,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        hidden_size=HIDDEN_SIZE,
    )
    assert isinstance(model, list)
    assert len(model) == virtual_pipeline_model_parallel_size
    optimizer = torch.optim.Adam(
        _get_params_for_weight_decay_optimization(model))

    initial_local_minibatch_size = get_num_microbatches() * args.micro_batch_size
    dataset = Dataset(NUM_SAMPLES)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=BatchSamplerCls(
            NUM_SAMPLES,
            0,
            initial_local_minibatch_size,
            parallel_state.get_data_parallel_rank(),
            parallel_state.get_data_parallel_world_size(),
        ),
    )
    data_iter = iter(data_loader)

    def get_num_samples(batch):
        if isinstance(batch, torch.Tensor):
            return len(batch)
        assert isinstance(batch, (list, tuple))
        return [get_num_samples(b) for b in batch]

    tensor_shape = [args.micro_batch_size, HIDDEN_SIZE, HIDDEN_SIZE]
    consumed_samples = 0
    for i in range(NUM_ITERATIONS):
        update_num_microbatches(consumed_samples, consistency_check=False)
        local_batch_size = get_num_microbatches() * args.micro_batch_size
        data_iter._index_sampler.local_minibatch_size = local_batch_size
        local_mini_batch = next(data_iter)

        _logger.info(
            f"iter: {i} / {NUM_ITERATIONS} "
            f"local batchsize: {get_num_samples(local_mini_batch)} "
            f"consumed_samples: {consumed_samples} / {NUM_SAMPLES}"
        )
        _forward_backward_pipelining_with_interleaving(
            fwd_step_func,
            local_mini_batch,
            model,
            forward_only=forward_only,
            tensor_shape=tensor_shape,
        )

        consumed_samples += (
            parallel_state.get_data_parallel_world_size()
            * get_num_microbatches()
            * args.micro_batch_size
        )

        if not forward_only:
            for m in model:
                for p in m.parameters():
                    if p.grad is None:
                        raise RuntimeError("grad not found")
            else:
                optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()


class DynamicBatchsizeTestBase:
    @unittest.skipUnless(torch.cuda.device_count() > 2, "requires at least 3 gpus")
    def test_dynamic_batchsize(self):

        n_tests = 0
        failures = []

        override_args = {
            "micro_batch_size": 2,
            "num_layers": 16,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "max_position_embeddings": 512,
            "seq_length": 512,
            "global_batch_size": 128,
            "use_cpu_initialization": True,
            "world_size": self.world_size,
            "rank": self.rank,
        }

        global_vars.set_global_variables(
            args_defaults={"global_batch_size": 512,
                           "rampup_batch_size": [64, 64, 1000], },
            ignore_unknown_args=True,
            override_args=override_args,
        )

        args = global_vars.get_args()

        setup_microbatch_calculator(
            args.rank,
            args.rampup_batch_size,
            args.global_batch_size,
            args.micro_batch_size,
            1,  # args.data_parallel_size,
        )
        for BatchSamplerCls in (
            MegatronPretrainingSampler,
            MegatronPretrainingRandomSampler,
        ):
            for forward_only in (False, True):
                n_tests += 1
                pipeline_model_parallel_size = self.world_size
                try:
                    run_interleaved_with_dynamic_batch_size(
                        pipeline_model_parallel_size, forward_only, BatchSamplerCls,
                    )
                except Exception as e:
                    msg = (
                        f"\tforward_only: {forward_only}\n"
                        f"pipeline rank: {parallel_state.get_pipeline_model_parallel_rank()}, "
                        f"virtual pipeline rank: {parallel_state.get_virtual_pipeline_model_parallel_rank()}\n"
                        f"{str(e)}"
                    )
                    raise RuntimeError(msg)
                finally:
                    parallel_state.destroy_model_parallel()
        if failures:
            print_separator("TEST FAILED:")
            print("\n".join(failures))
            msg = f"{len(failures)} / {n_tests} cases failed"
            raise RuntimeError(msg)
        else:
            if torch.distributed.get_rank() == 0:
                print_separator("TEST RESULT: ### PASS!")


class NcclDynamicBatchsizeTest(DynamicBatchsizeTestBase, NcclDistributedTestBase):
    pass

# TODO: (Fuzzkatt) UCC still doesn't work with fwd_bwd_pipelining_with_interleaving


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    common_utils.run_tests()
