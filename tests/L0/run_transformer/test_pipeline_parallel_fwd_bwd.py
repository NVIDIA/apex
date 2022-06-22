import logging
import itertools
import re
from typing import Optional, Tuple, List
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal import common_cuda

logging.getLogger("torch").setLevel(logging.WARNING)

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
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase
from apex.transformer.testing.distributed_test_base import HAS_TORCH_UCC
from apex.transformer.testing.distributed_test_base import HAS_TORCH_UCC_COMPAT_NVIDIA_DRIVER
from apex.transformer.testing import commons as testing_utils

logging.getLogger("apex").setLevel(logging.WARNING)

weight_coeff = 1024


def get_init_weights_func(offset: int = 0):
    @torch.no_grad()
    def init_weights(m):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        if isinstance(m, torch.nn.Linear):
            m.weight.fill_((rank + offset + 1.0) / weight_coeff)
            m.bias.fill_(1.0)
    return init_weights


def get_target_loss_and_model(global_batch_shape: tuple, hidden_size: int, total_layers: int) -> Tuple[torch.Tensor, List[torch.Tensor]]: 
    model = []
    data = torch.ones(global_batch_shape, dtype=torch.double)
    for i in range(total_layers):
        w = torch.ones((hidden_size, hidden_size), dtype=torch.double) * (i + 1.0) / weight_coeff
        b = torch.ones(hidden_size, dtype=torch.double)

        w.requires_grad_()
        b.requires_grad_()

        # don't need to care about transpose semantics as all values are the same
        data = torch.matmul(w, data) + b
        model.append([w, b])

    loss = data.sum() / global_batch_shape[0]
    loss.backward()

    return loss, model


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


class PipelineParallelForwardBackwardTestBase:

    GLOBAL_BATCH_SIZE = 16
    MICRO_BATCH_SIZE = 2
    HIDDEN_SIZE = 32

    deallocate_options = (True, False)
    # If :obj:`None`, (torch.float32, torch.float16, torch.bfloat16) are dtype options on Ampere.
    # You can limit the options by overriding the following `dtypes`.
    dtypes = None

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
                model_module.apply(get_init_weights_func(idx*offset))

            _param_groups = _get_params_for_weight_decay_optimization(model)
            optimizer = torch.optim.Adam(_param_groups, lr=1e-3)

            pp_utils.update_num_microbatches(0)
            
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

            if dtype == torch.double:
                hidden_size = self.HIDDEN_SIZE
                microbatch_size = self.MICRO_BATCH_SIZE
                total_layers = pipeline_model_parallel_world_size
                if virtual_pipeline_model_parallel_size is not None:
                    total_layers *= virtual_pipeline_model_parallel_size
                target_loss, target_model = get_target_loss_and_model(global_batch_shape, hidden_size, total_layers)

                for loss_item in loss:
                    x = loss_item['avg']
                    torch.testing.assert_close(x.item() / microbatch_size, target_loss.item())

                if not forward_only:
                    for vm_id, model_module in enumerate(model):
                        params = list(model_module.parameters())
                        rank = params[0].get_device()
                        offset = pipeline_model_parallel_world_size
                        param_id = rank // data_parallel_size + vm_id * offset
                        target_params = target_model[param_id]

                        torch.testing.assert_close(params[0].cpu(), target_params[0])
                        torch.testing.assert_close(params[1].cpu(), target_params[1])
                        torch.testing.assert_close(params[0].grad.cpu() / microbatch_size, target_params[0].grad)
                        torch.testing.assert_close(params[1].grad.cpu() / microbatch_size, target_params[1].grad)

            if not forward_only:
                for m in model:
                    for p in m.parameters():
                        self.assertIsNotNone(p.grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            parallel_state.destroy_model_parallel()

    def test_no_pipelining(self):
        self._forward_backward_test_impl(False, forward_backward_no_pipelining, 1, None)

    def test_no_pipelining_inference(self):
        self._forward_backward_test_impl(True, forward_backward_no_pipelining, 1, None)

    def test_pipelining_without_interleaving(self):
        self._forward_backward_test_impl(
            False, forward_backward_pipelining_without_interleaving, None, None
        )

    def test_pipelining_async(self):
        self._forward_backward_test_impl(
            False, forward_backward_pipelining_without_interleaving, None, None, async_comm=True
        )

    def test_pipelining_without_interleaving_inference(self):
        self._forward_backward_test_impl(
            True, forward_backward_pipelining_without_interleaving, None, None
        )

    def test_pipelining_inference_async(self):
        self._forward_backward_test_impl(
            True, forward_backward_pipelining_without_interleaving, None, None, async_comm=True
        )

    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Megatron-LM voodoo")
    def test_pipelining_with_interleaving(self):
        self._forward_backward_test_impl(
            False, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2
        )

    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Megatron-LM voodoo")
    def test_pipelining_with_interleaving_inference(self):
        self._forward_backward_test_impl(
            True, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2
        )


class NcclPipelineParallelForwardBackwardTest(NcclDistributedTestBase, PipelineParallelForwardBackwardTestBase):

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def _run_hybrid_distributed_backend(self, forward_only: bool) -> None:
        self._forward_backward_test_impl(
            forward_only, forward_backward_pipelining_without_interleaving, None, None,
            default_backend="nccl", p2p_backend="ucc",
        )

    @unittest.skipUnless(HAS_TORCH_UCC_COMPAT_NVIDIA_DRIVER, "Needs driver >= 470.42.01")
    def _test_hybrid_backends(self, forward_only: bool) -> None:
        if HAS_TORCH_UCC:
            self._run_hybrid_distributed_backend(forward_only)
        else:
            with self.assertRaisesRegex(
                ImportError,
                re.escape("UCC backend requires [torch_ucc](https://github.com/facebookresearch/torch_ucc) but not found"),
            ):
                self._run_hybrid_distributed_backend(forward_only)

    def test_pipelining_without_interleaving_ucc_for_p2p(self):
        self._test_hybrid_backends(False)

    def test_pipelining_without_interleaving_inference_ucc_for_p2p(self):
        self._test_hybrid_backends(True)


# n.b.(mkozuki): pipeline parallel w/o interleaving with UCX_TLS=tcp,sm fails.
class UccPipelineParallelForwardBackwardTest(UccDistributedTestBase, PipelineParallelForwardBackwardTestBase):

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    deallocate_options = (False,)
    dtypes = (torch.float32,)


if __name__ == "__main__":
    common_utils.run_tests()
