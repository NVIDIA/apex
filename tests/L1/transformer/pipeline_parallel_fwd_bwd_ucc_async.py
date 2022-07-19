import os
import logging
import itertools
from typing import Optional, Tuple, List
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal import common_cuda
from torch.testing._internal import common_distributed

from apex._autocast_utils import _get_autocast_dtypes
from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import utils as pp_utils
from apex.transformer.pipeline_parallel.schedules.common import (
    FwdStepFunc,
    build_model,
    _get_params_for_weight_decay_optimization,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import (
    forward_backward_no_pipelining,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import (
    _forward_backward_pipelining_with_interleaving,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase
from apex.transformer.testing import commons as testing_utils


logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("apex").setLevel(logging.WARNING)


def _get_default_world_sizes_model_parallel_world_size(pipeline_model_parallel_world_size: Optional[int] = None
    ) -> Tuple[int, int, int]:
    # TODO: revisit if we can fold this into the class for skip logic / avoid duplication
    # of world size computation
    world_size = torch.cuda.device_count()
    tensor_model_parallel_world_size = 1
    data_parallel_size = 1 + (world_size >= 8 and world_size % 2 == 0)

    if pipeline_model_parallel_world_size is None:
        pipeline_model_parallel_world_size =  world_size // (tensor_model_parallel_world_size * data_parallel_size)
    else:
        data_parallel_size = world_size // (tensor_model_parallel_world_size * pipeline_model_parallel_world_size)

    return tensor_model_parallel_world_size, data_parallel_size, pipeline_model_parallel_world_size


class UccPipelineParallelForwardBackwardProf(UccDistributedTestBase):

    # The purpose of this class is to test and confirm asynchronous communication via profiling.
    # Having that in mind, it is safe to skip all the numerical checks.
    # For unit testing with numerical checks please refer to `tests/L0/run_transformer/test_pipeline_parallel_fwd_bwd.py`.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.GLOBAL_BATCH_SIZE = 1024
        self.MICRO_BATCH_SIZE = 64
        self.HIDDEN_SIZE = 256
        self.NUM_FWD_BWD_ITERATIONS = 4
        self.deallocate_options = (False,)
        self.dtypes = (torch.float32,)

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def _forward_backward_test_impl(
        self,
        forward_only: bool,
        fwd_bwd_func: FwdStepFunc,
        pipeline_model_parallel_world_size: Optional[int],
        virtual_pipeline_model_parallel_size: Optional[int],
        async_comm: bool = False,
        *,
        default_backend: Optional[str] = None,
        p2p_backend: Optional[str] = None,
    ) -> None:
        if fwd_bwd_func == _forward_backward_pipelining_with_interleaving:
            self.assertIsNotNone(virtual_pipeline_model_parallel_size)
            self.assertGreater(virtual_pipeline_model_parallel_size, 1)
        dtype_options = self.dtypes or [torch.float32, torch.double] + _get_autocast_dtypes()

        for dtype, deallocate_pipeline_outputs in itertools.product(
            dtype_options, self.deallocate_options,
        ):
            grad_scaler = (
                torch.cuda.amp.GradScaler(init_scale=4.0)
                if dtype == torch.half
                else None
            )

            (tensor_model_parallel_world_size,
            data_parallel_size,
            pipeline_model_parallel_world_size) = _get_default_world_sizes_model_parallel_world_size(pipeline_model_parallel_world_size)

            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size_=tensor_model_parallel_world_size,
                pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
                virtual_pipeline_model_parallel_size_=virtual_pipeline_model_parallel_size,
                default_backend=default_backend,
                p2p_backend=p2p_backend,
            )
            pp_utils._reconfigure_microbatch_calculator(
                rank=parallel_state.get_tensor_model_parallel_rank(),
                rampup_batch_size=None,
                global_batch_size=self.GLOBAL_BATCH_SIZE,
                micro_batch_size=self.MICRO_BATCH_SIZE,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

            global_batch_shape = (
                self.GLOBAL_BATCH_SIZE
                // parallel_state.get_data_parallel_world_size(),
                self.HIDDEN_SIZE,
                self.HIDDEN_SIZE,
            )

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (torch.ones(global_batch_shape, dtype=dtype).cuda(), )

            model = build_model(
                testing_utils.model_provider_func,
                # Use DDP only when it's better to have
                wrap_with_ddp=data_parallel_size > 1,
                virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
                hidden_size=self.HIDDEN_SIZE,
            )


            offset = pipeline_model_parallel_world_size if virtual_pipeline_model_parallel_size is not None else 0
            for idx, model_module in enumerate(model):
                model_module = model_module.to(dtype)

            _param_groups = _get_params_for_weight_decay_optimization(model)
            optimizer = torch.optim.Adam(_param_groups, lr=1e-3)

            pp_utils.update_num_microbatches(0)

            for _ in range(self.NUM_FWD_BWD_ITERATIONS):
                loss = fwd_bwd_func(
                    testing_utils.fwd_step_func,
                    batch,
                    model,
                    forward_only=forward_only,
                    # `tensor_shape` is the shape of micro batch.
                    tensor_shape=(
                        self.MICRO_BATCH_SIZE,
                        self.HIDDEN_SIZE,
                        self.HIDDEN_SIZE,
                    ),
                    dtype=dtype,
                    async_comm=async_comm,
                    grad_scaler=grad_scaler,
                    deallocate_pipeline_output=deallocate_pipeline_outputs,
                )

            parallel_state.destroy_model_parallel()

    def test_learning_no_pipelining(self):
        self._forward_backward_test_impl(False, forward_backward_no_pipelining, 1, None)

    def test_inference_no_pipelining(self):
        self._forward_backward_test_impl(True, forward_backward_no_pipelining, 1, None)

    def test_learning_pipelining_without_interleaving(self):
        self._forward_backward_test_impl(
            False, forward_backward_pipelining_without_interleaving, None, None
        )

    def test_inference_pipelining_without_interleaving(self):
        self._forward_backward_test_impl(
            True, forward_backward_pipelining_without_interleaving, None, None
        )

    def test_learning_async_pipelining_without_interleaving(self):
        self._forward_backward_test_impl(
            False, forward_backward_pipelining_without_interleaving, None, None, async_comm=True
        )

    def test_inference_async_pipelining_without_interleaving(self):
        self._forward_backward_test_impl(
            True, forward_backward_pipelining_without_interleaving, None, None, async_comm=True
        )

    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_learning_pipelining_with_interleaving(self):
        self._forward_backward_test_impl(
            False, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2
        )

    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_inference_pipelining_with_interleaving(self):
        self._forward_backward_test_impl(
            True, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2
        )

    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_learning_async_pipelining_with_interleaving(self):
        self._forward_backward_test_impl(
            False, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2, async_comm=True
        )

    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_inference_async_pipelining_with_interleaving(self):
        self._forward_backward_test_impl(
            True, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2, async_comm=True
        )


if __name__ == "__main__":
    os.environ["UCC_TLS"] = "ucp,cuda"
    common_distributed.TIMEOUT_DEFAULT = 500
    common_utils.run_tests()
