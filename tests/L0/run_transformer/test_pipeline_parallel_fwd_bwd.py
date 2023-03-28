import contextlib
import logging
import itertools
import os
from packaging.version import parse, Version
import re
from typing import Optional, Tuple, List
import unittest

import torch
from torch.testing._internal import common_utils

from apex._autocast_utils import _get_autocast_dtypes
from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
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
from apex.transformer.testing.distributed_test_base import HAS_TORCH_UCC_COMPAT_NVIDIA_DRIVER
from apex.transformer.testing import commons as testing_utils
from apex.transformer._ucc_util import HAS_UCC

logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("apex").setLevel(logging.WARNING)

weight_coeff = 1024

# Guard for https://github.com/pytorch/pytorch/pull/82450
CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV = False
ngc_container_2209, pytorch_113 = Version("22.09"), Version("1.13")
if parse(os.getenv("NVIDIA_PYTORCH_VERSION", "22.08")) >= ngc_container_2209:
    CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV = True
elif parse(torch.__version__) >= pytorch_113:
    CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV = True
else:
    CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV = False


def get_init_weights_func(offset: int = 0):
    @torch.no_grad()
    def init_weights(m):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        if isinstance(m, torch.nn.Linear):
            m.weight.fill_((rank + offset + 1.0) / weight_coeff)
            m.bias.fill_(1.0)
    return init_weights


def get_dtype_for_comparison():
    if(torch.cuda.get_device_capability() >= (8, 0)):
        return torch.float64
    return torch.float32


def get_target_loss_and_model(global_batch_shape: tuple, hidden_size: int, total_layers: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    model = []
    dtype = get_dtype_for_comparison()
    data = torch.ones(global_batch_shape, dtype=dtype)
    for i in range(total_layers):
        w = torch.ones((hidden_size, hidden_size), dtype=dtype) * (i + 1.0) / weight_coeff
        b = torch.ones(hidden_size, dtype=dtype)

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
        sync_batch_comm: bool = True,
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
                sync_batch_comm=sync_batch_comm,
            )

            if dtype == get_dtype_for_comparison():
                torch.cuda.synchronize()
                hidden_size = self.HIDDEN_SIZE
                microbatch_size = self.MICRO_BATCH_SIZE
                total_layers = pipeline_model_parallel_world_size
                if virtual_pipeline_model_parallel_size is not None:
                    total_layers *= virtual_pipeline_model_parallel_size
                target_loss, target_model = get_target_loss_and_model(global_batch_shape, hidden_size, total_layers)

                for loss_item in loss:
                    x = loss_item['avg']
                    self.assertEqual(x.item() / microbatch_size, target_loss.item())

                if not forward_only:
                    for vm_id, model_module in enumerate(model):
                        params = list(model_module.parameters())
                        rank = params[0].get_device()
                        offset = pipeline_model_parallel_world_size
                        param_id = rank // data_parallel_size + vm_id * offset
                        target_params = target_model[param_id]

                        self.assertEqual(params[0].cpu(), target_params[0])
                        self.assertEqual(params[1].cpu(), target_params[1])
                        self.assertEqual(params[0].grad.cpu() / microbatch_size, target_params[0].grad)
                        self.assertEqual(params[1].grad.cpu() / microbatch_size, target_params[1].grad)

            if not forward_only:
                for m in model:
                    for p in m.parameters():
                        self.assertIsNotNone(p.grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            parallel_state.destroy_model_parallel()

    def test_learning_no_pipelining(self):
        self._forward_backward_test_impl(False, forward_backward_no_pipelining, 1, None)

    def test_inference_no_pipelining(self):
        self._forward_backward_test_impl(True, forward_backward_no_pipelining, 1, None)

    def test_learning_pipelining_without_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            False, forward_backward_pipelining_without_interleaving, None, None, sync_batch_comm=sync_batch_comm,
        )

    def test_inference_pipelining_without_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            True, forward_backward_pipelining_without_interleaving, None, None, sync_batch_comm=sync_batch_comm,
        )

    def test_learning_async_pipelining_without_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            False, forward_backward_pipelining_without_interleaving, None, None, async_comm=True,
            sync_batch_comm=sync_batch_comm,
        )

    def test_inference_async_pipelining_without_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            True, forward_backward_pipelining_without_interleaving, None, None, async_comm=True,
            sync_batch_comm=sync_batch_comm,
        )

    # fails on native ucc: times out
    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_learning_pipelining_with_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            False, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2,
            sync_batch_comm=sync_batch_comm,
        )

    # fails on native ucc: times out
    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_inference_pipelining_with_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            True, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2,
            sync_batch_comm=sync_batch_comm,
        )

    # fails on native ucc: times out
    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_learning_async_pipelining_with_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            False, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2, async_comm=True,
            sync_batch_comm=sync_batch_comm,
        )

    # fails on native ucc: times out
    @unittest.skipUnless(_get_default_world_sizes_model_parallel_world_size()[-1] > 2, "Interleaved schedule requires pipeline_model_parallel_world_size > 2")
    def test_inference_async_pipelining_with_interleaving(self, sync_batch_comm: bool = True):
        self._forward_backward_test_impl(
            True, _forward_backward_pipelining_with_interleaving, None, virtual_pipeline_model_parallel_size=2, async_comm=True,
            sync_batch_comm=sync_batch_comm,
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
        if HAS_UCC:
            self._run_hybrid_distributed_backend(forward_only)
        else:
            with self.assertRaisesRegex(
                ImportError,
                re.escape("UCC backend requires pytorch source build with UCC installed and enabled"),
            ):
                self._run_hybrid_distributed_backend(forward_only)

    def test_learning_pipelining_without_interleaving_ucc_for_p2p(self):
        self._test_hybrid_backends(False)

    def test_inference_pipelining_without_interleaving_ucc_for_p2p(self):
        self._test_hybrid_backends(True)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_learning_pipelining_without_interleaving_skyp_sync_after_batch_isend_irecv(self):
        self.test_learning_pipelining_without_interleaving(sync_batch_comm=False)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_inference_pipelining_without_interleaving_skip_sync_after_batch_isend_irecv(self):
        self.test_inference_pipelining_without_interleaving(sync_batch_comm=False)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_learning_async_pipelining_without_interleaving_skip_sync_after_batch_isend_irecv(self):
        self.test_learning_async_pipelining_without_interleaving(sync_batch_comm=False)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_inference_async_pipelining_without_interleaving_skip_sync_after_batch_isend_irecv(self):
        self.test_inference_async_pipelining_without_interleaving(sync_batch_comm=False)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_learning_pipelining_with_interleaving_skip_sync_after_batch_isend_irecv(self):
        self.test_learning_pipelining_with_interleaving(sync_batch_comm=False)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_inference_pipelining_with_interleaving_skip_sync_after_batch_isend_irecv(self):
        self.test_inference_pipelining_with_interleaving(sync_batch_comm=False)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_learning_async_pipelining_with_interleaving_skip_sync_after_batch_isend_irecv(self):
        self.test_learning_async_pipelining_with_interleaving(sync_batch_comm=False)

    @unittest.skipUnless(CAN_SKIP_SYNC_AFTER_BATCH_ISEND_IRECV, "Requires https://github.com/pytorch/pytorch/pull/82450")
    def test_inference_async_pipelining_with_interleaving_skip_sync_after_batch_isend_irecv(self):
        self.test_inference_async_pipelining_with_interleaving(sync_batch_comm=False)


# n.b.(mkozuki): pipeline parallel w/o interleaving with UCX_TLS=tcp,sm fails.
class UccPipelineParallelForwardBackwardTest(UccDistributedTestBase, PipelineParallelForwardBackwardTestBase):

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    deallocate_options = (False,)
    dtypes = (torch.float32,)


# Sanity checking the functionality of `forward_backward_pipelining_without_interleaving` with
# `model_type=ModelType.encoder_and_decoder` which is used for pipeline training of transformer
# models such as T5.
@unittest.skipIf(torch.cuda.device_count() < 4, "Requires >= 4 GPUs")
class NcclPipelineParallelWithToyParallelMLP(NcclDistributedTestBase):

    GLOBAL_BATCH_SIZE: int = 16
    MICRO_BATCH_SIZE: int = 2
    HIDDEN_SIZE: int = 64
    # TODO(mkozuki): Change `DECODER_SEQUENCE_LENGTH` to a value different from `ENCODER_SEQUENCE_LENGTH`.
    # To test forward_backward_pipelining_without_interleaving with `model_type=ModelType.encoder_and_decoder`,
    # `decoder_seq_length` is necessary and ideally should be different from `encoder_sequence_length`
    # but my laziness let me use the same value.
    # Note that you may have to either update `MyModel` def or define another `MyModel`.
    # to support different `DECODER_SEQUENCE_LENGTH`.
    ENCODER_SEQUENCE_LENGTH: int = 32
    DECODER_SEQUENCE_LENGTH: int = 32

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    # TODO(mkozuki): Set `tensor_model_parallel>1` for encoder_and_decoder as well if there's enough GPUs
    #   in order to let `sequence_parallel_enabled` have an effect on tensor shape logic.
    def _forward_backward_test_impl(
        self,
        *,
        forward_only: bool,
        sequence_parallel_enabled: bool,
        model_type: ModelType,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # N.B.(mkozuki): It might be better to set `tensor_model_parallel_size` to >1
        # if `self.world_size > 5`. Otherwise, `pipeline_model_parallel_split_rank`
        # can be 1, which can be too far real usecase.
        tensor_model_parallel_size = 1 + int(self.world_size >= 4)
        pipeline_model_parallel_world_size = self.world_size // tensor_model_parallel_size
        if model_type == ModelType.encoder_and_decoder:
            pipeline_model_parallel_split_rank = pipeline_model_parallel_world_size // 2
        else:
            pipeline_model_parallel_split_rank = None

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=tensor_model_parallel_size,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            virtual_pipeline_model_parallel_size_=None,
            pipeline_model_parallel_split_rank_=pipeline_model_parallel_split_rank,
        )
        testing_utils.set_random_seed(567)
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        # TODO(mkozuki): Call `build_model` with `model_type`.
        model = build_model(
            testing_utils.mlp_provider_func,
            wrap_with_ddp=False,
            virtual_pipeline_model_parallel_size=None,
            hidden_size=self.HIDDEN_SIZE,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )
        model = [m.to(dtype=dtype) for m in model]

        if parallel_state.is_pipeline_first_stage():
            batch: Tuple[torch.Tensor] = (
                torch.ones(
                    (self.GLOBAL_BATCH_SIZE, self.ENCODER_SEQUENCE_LENGTH, self.HIDDEN_SIZE),
                    dtype=dtype,
                    device="cuda",
                ),
            )
        else:
            batch = None

        forward_backward_pipelining_without_interleaving(
            forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                sequence_parallel_enabled=sequence_parallel_enabled,
            ),
            batch=batch,
            model=model,
            forward_only=forward_only,
            tensor_shape=(
                self.ENCODER_SEQUENCE_LENGTH,
                self.MICRO_BATCH_SIZE,
                self.HIDDEN_SIZE,
            ),
            model_type=model_type,
            decoder_sequence_length=self.DECODER_SEQUENCE_LENGTH,
            async_comm=False,
            grad_scaler=None,
            deallocate_pipeline_outputs=False,
            dtype=dtype,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    def test_pipelining_without_interleaving_encoder_and_decoder(self) -> None:
        self._forward_backward_test_impl(forward_only=False, sequence_parallel_enabled=False, model_type=ModelType.encoder_and_decoder)

    def test_pipelining_without_interleaving_inferenc_encoder_and_decoder(self) -> None:
        self._forward_backward_test_impl(forward_only=True, sequence_parallel_enabled=False, model_type=ModelType.encoder_and_decoder)

    def test_pipelining_without_interleaving_sequence_paralle_encoder_and_decoder(self) -> None:
        self._forward_backward_test_impl(forward_only=False, sequence_parallel_enabled=True, model_type=ModelType.encoder_and_decoder)

    def test_pipelining_without_interleaving_inference_sequence_paralle_encoder_and_decoder(self) -> None:
        self._forward_backward_test_impl(forward_only=True, sequence_parallel_enabled=True, model_type=ModelType.encoder_and_decoder)

    def test_pipelining_without_interleaving_encoder_or_decoder(self) -> None:
        self._forward_backward_test_impl(forward_only=False, sequence_parallel_enabled=False, model_type=ModelType.encoder_or_decoder)

    def test_pipelining_without_interleaving_sequence_parallel_encoder_or_decoder(self) -> None:
        self._forward_backward_test_impl(forward_only=False, sequence_parallel_enabled=True, model_type=ModelType.encoder_or_decoder)

    def test_pipelining_without_interleaving_sequence_parallel_encoder_or_decoder_half(self) -> None:
        self._forward_backward_test_impl(forward_only=False, sequence_parallel_enabled=True, model_type=ModelType.encoder_or_decoder, dtype=torch.half)


class NcclPipelineParallelWithCustomSyncContextHandler(NcclDistributedTestBase):

    GLOBAL_BATCH_SIZE = 32
    MICRO_BATCH_SIZE = 1
    HIDDEN_SIZE = 1

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    @unittest.skipIf(torch.cuda.device_count() < 2 or torch.cuda.device_count() % 2 != 0, "Requires >= 2 GPUs")
    def test_pipelining_without_interleaving_with_custom_sync_context_handler(self) -> None:

        # Parallel configuration
        world_size = torch.cuda.device_count()
        tensor_model_parallel_world_size = 1
        data_parallel_size = 2 if world_size > 2 else 1
        pipeline_model_parallel_world_size = world_size // data_parallel_size

        # Initialize pipeline parallelism
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=tensor_model_parallel_world_size,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        pp_utils.update_num_microbatches(0)

        # Construct synthetic data
        dtype = get_dtype_for_comparison()
        hidden_size = self.HIDDEN_SIZE
        microbatch_size = self.MICRO_BATCH_SIZE
        global_batch_shape = (
            self.GLOBAL_BATCH_SIZE
            // parallel_state.get_data_parallel_world_size(),
            hidden_size,
            hidden_size,
        )
        batch = None
        if parallel_state.is_pipeline_first_stage():
            batch = (torch.ones(global_batch_shape, dtype=dtype).cuda(), )

        # Construct model
        model = build_model(
            testing_utils.model_provider_func,
            wrap_with_ddp=True,
            hidden_size=hidden_size,
        )[0]
        model = model.to(dtype)
        model.module.apply(get_init_weights_func(0))

        # Construct context that destroys all grads on exit
        has_entered_grad_sync_context = False
        has_exited_grad_sync_context = False
        has_called_grad_sync_func = False
        @contextlib.contextmanager
        def custom_grad_sync_context():
            try:
                nonlocal has_entered_grad_sync_context
                has_entered_grad_sync_context = True
                yield
            finally:
                nonlocal has_exited_grad_sync_context
                has_exited_grad_sync_context = True
                for param in model.parameters():
                    param.grad = None
        def custom_grad_sync_func():
            nonlocal has_called_grad_sync_func
            has_called_grad_sync_func = True

        # Training step with pipeline parallelism
        loss = forward_backward_pipelining_without_interleaving(
            testing_utils.fwd_step_func,
            batch,
            model,
            forward_only=False,
            tensor_shape=(microbatch_size, hidden_size, hidden_size),
            dtype=dtype,
            async_comm=False,
            grad_scaler=None,
            deallocate_pipeline_outputs=False,
            sequence_parallel_enabled=False,
            custom_sync_context_handler=custom_grad_sync_context,
            custom_grad_sync_func=custom_grad_sync_func,
        )
        torch.cuda.synchronize()

        # Check if model has initialized gradients
        has_any_grads = any(param.grad is not None for param in model.parameters())
        has_all_grads = all(param.grad is not None for param in model.parameters())

        # Check context behavior
        self.assertTrue(has_entered_grad_sync_context, 'Has not entered custom sync context')
        self.assertTrue(has_exited_grad_sync_context, 'Has not exited custom sync context')
        self.assertEqual(
            has_any_grads,
            has_all_grads,
            'Expected gradients to all be uninitialized or all be initialized',
        )
        self.assertEqual(
            has_all_grads,
            parallel_state.is_pipeline_first_stage(),
            'Expected gradients to be initialized only in first pipeline stage',
        )

        # Clean up
        parallel_state.destroy_model_parallel()

    @unittest.skipIf(torch.cuda.device_count() < 4 or torch.cuda.device_count() % 2 != 0, "Requires >= 4 GPUs")
    def test_pipelining_with_interleaving_with_custom_sync_context_handler(self) -> None:

        # Parallel configuration
        world_size = torch.cuda.device_count()
        tensor_model_parallel_world_size = 1
        data_parallel_size = 2 if world_size > 4 else 1
        pipeline_model_parallel_world_size = world_size // data_parallel_size
        virtual_pipeline_model_parallel_size = 2

        # Initialize pipeline parallelism
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=tensor_model_parallel_world_size,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            virtual_pipeline_model_parallel_size_=virtual_pipeline_model_parallel_size,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        pp_utils.update_num_microbatches(0)

        # Construct synthetic data
        dtype = get_dtype_for_comparison()
        hidden_size = self.HIDDEN_SIZE
        microbatch_size = self.MICRO_BATCH_SIZE
        global_batch_shape = (
            self.GLOBAL_BATCH_SIZE
            // parallel_state.get_data_parallel_world_size(),
            hidden_size,
            hidden_size,
        )
        batch = None
        if parallel_state.is_pipeline_first_stage():
            batch = (torch.ones(global_batch_shape, dtype=dtype).cuda(), )

        # Construct model
        model = build_model(
            testing_utils.model_provider_func,
            wrap_with_ddp=True,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            hidden_size=hidden_size,
        )
        for module in model:
            module.to(dtype)
            module.module.apply(get_init_weights_func(0))

        # Construct context that keeps track whenever entered/exited
        grad_sync_context_enter_count = 0
        grad_sync_context_exit_count = 0
        @contextlib.contextmanager
        def custom_grad_sync_context():
            try:
                nonlocal grad_sync_context_enter_count
                grad_sync_context_enter_count += 1
                yield
            finally:
                nonlocal grad_sync_context_exit_count
                grad_sync_context_exit_count += 1
                for module in model:
                    for param in module.parameters():
                        param.grad = None

        # Training step with pipeline parallelism
        loss = _forward_backward_pipelining_with_interleaving(
            testing_utils.fwd_step_func,
            batch,
            model,
            forward_only=False,
            tensor_shape=(microbatch_size, hidden_size, hidden_size),
            dtype=dtype,
            async_comm=False,
            grad_scaler=None,
            deallocate_pipeline_outputs=False,
            sequence_parallel_enabled=False,
            custom_sync_context_handler=custom_grad_sync_context,
        )
        torch.cuda.synchronize()

        # Check context behavior
        self.assertTrue(
            grad_sync_context_enter_count > 0,
            'Has not entered custom sync context',
        )
        self.assertEqual(
            grad_sync_context_enter_count,
            grad_sync_context_exit_count,
            'Has not entered and exited custom sync context '
            'the same number of times',
        )
        self.assertEqual(
            grad_sync_context_exit_count,
            virtual_pipeline_model_parallel_size + 1,
            'Expected to exit custom sync context once per model chunk '
            'and once at the function end',
        )

        # Clean up
        parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    common_utils.run_tests()
