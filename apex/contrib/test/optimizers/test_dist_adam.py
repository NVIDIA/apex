from contextlib import contextmanager
import io
from typing import Callable, Optional, Tuple
import unittest
import warnings

import torch
from torch.testing._internal import common_utils

SKIP_TEST = None
try:
    from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
except ImportError as e:
    SKIP_TEST = e
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase


class SimpleModel(torch.nn.Module):
    def __init__(self, num_layers, size):
        super().__init__()
        self.params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(1, size) + 1)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        y = 0
        for i, param in enumerate(self.params):
            y += (i+1) * param * x
        return y


def make_models(
        num_layers: int,
        size: int,
        *,
        lr: float = 0.1,
        adam_w_mode: bool = True,
        model_dtype: torch.dtype = torch.float32,
        optim_dtype: Optional[torch.dtype] = None,
        grad_sync_dtype: Optional[torch.dtype] = None,
        param_sync_dtype: Optional[torch.dtype] = None,
        device: torch.device = 'cuda',
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        average_grad_sync: bool = True,
        overlap_communication: bool = True,
        bucket_cap_mb: float = 71/(4*1024*1024),
        contiguous_buffers: bool = False,
        store_params: bool = False,
        store_param_remainders: bool = False,
        with_scaled_states: bool = False,
        nccl_ub: bool = False,
):

    # Construct models with same parameters
    ref_model = SimpleModel(num_layers, size).to(dtype=model_dtype, device=device)
    dist_model = SimpleModel(num_layers, size).to(dtype=model_dtype, device=device)
    with torch.no_grad():
        for ref_param, dist_param in zip(dist_model.parameters(),
                                         ref_model.parameters()):
            dist_param.copy_(ref_param)

    # Initialize reference model with data-parallelism
    rank = torch.distributed.get_rank()
    ref_model = torch.nn.parallel.DistributedDataParallel(
        ref_model,
        device_ids=[rank] if device=='cuda' else None,
        output_device=rank if device=='cuda' else None,
        process_group=process_group,
    )

    # Construct optimizers with same hyperparameters
    if optim_dtype is None:
        optim_dtype = model_dtype
    optim_args = dict(lr=lr, betas=(0.1,0.2), eps=0.25, weight_decay=0.1)
    ref_optim_class = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    ref_optim = ref_optim_class(
        [
            {'params': list(ref_model.parameters())[1::2], 'lr': lr*2},
            {'params': list(ref_model.parameters())[0::2]},
        ],
        **optim_args,
    )
    dist_optim = DistributedFusedAdam(
        [
            {'params': list(dist_model.parameters())[1::2], 'lr': lr*2},
            {'params': list(dist_model.parameters())[0::2]},
        ],
        adam_w_mode=adam_w_mode,
        overlap_grad_sync=overlap_communication,
        overlap_param_sync=overlap_communication,
        bucket_cap_mb=bucket_cap_mb,
        dtype=optim_dtype,
        grad_sync_dtype=grad_sync_dtype,
        param_sync_dtype=param_sync_dtype,
        process_group=process_group,
        average_grad_sync=average_grad_sync,
        contiguous_param_buffer=contiguous_buffers,
        contiguous_grad_buffer=contiguous_buffers,
        store_params=store_params,
        store_param_remainders=store_param_remainders,
        with_scaled_states=with_scaled_states,
        nccl_ub=nccl_ub,
        **optim_args,
    )

    return ref_model, ref_optim, dist_model, dist_optim


@contextmanager
def dummy_context():
    try:
        yield
    finally:
        pass


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class TestDistributedFusedAdam(NcclDistributedTestBase):

    seed = 1234

    def test_matches_pytorch(
            self,
            rtol: Optional[float] = None,
            atol: Optional[float] = None,
            num_layers: int = 11,
            layer_size: int = 7,
            batch_size: int = 3,
            num_steps: int = 3,
            micro_batch_steps: int = 3,
            adam_w_mode: bool = True,
            overlap_communication: bool = True,
            use_nosync: bool = True,
            model_dtype: torch.dtype = torch.float32,
            optim_dtype: Optional[torch.dtype] = None,
            grad_sync_dtype: Optional[torch.dtype] = None,
            param_sync_dtype: Optional[torch.dtype] = None,
            device: torch.device = 'cuda',
            bucket_cap_mb: float = 71/(4*1024*1024),
            contiguous_buffers: bool = False,
            store_params: bool = False,
            store_param_remainders: bool = False,
            with_scaled_states: bool = False,
            nccl_ub: bool = False,
            init_optim_func: Optional[Callable[[DistributedFusedAdam], None]] = None,
    ):

        torch.manual_seed(self.seed + self.rank)

        # Identical models with data-parallel and ZeRO
        ref_model, ref_optim, dist_model, dist_optim = make_models(
            num_layers,
            layer_size,
            adam_w_mode=adam_w_mode,
            model_dtype=model_dtype,
            optim_dtype=optim_dtype,
            grad_sync_dtype=grad_sync_dtype,
            param_sync_dtype=param_sync_dtype,
            device=device,
            overlap_communication=overlap_communication,
            bucket_cap_mb=bucket_cap_mb,
            contiguous_buffers=contiguous_buffers,
            store_params=store_params,
            store_param_remainders=store_param_remainders,
            with_scaled_states=with_scaled_states,
            nccl_ub=nccl_ub,
        )

        # Initialize distributed optimizer
        if init_optim_func is not None:
            init_optim_func(dist_optim)

        # Training loop
        for step in range(num_steps):

            # Reset gradients
            ref_optim.zero_grad()
            dist_optim.zero_grad()

            # Forward and backward passes
            for micro_step in range(micro_batch_steps):

                # Synthetic data
                x = torch.rand(batch_size, layer_size) - 0.5
                dy = torch.rand_like(x) - 0.5
                x = x.to(dtype=model_dtype, device=device)
                dy = dy.to(dtype=model_dtype, device=device)

                # Reference implementation
                x_ref = x.detach().clone().requires_grad_(True)
                y_ref = ref_model(x_ref)
                y_ref.backward(dy)

                # Distributed implementation
                x_dist = x.detach().clone().requires_grad_(True)
                y_dist = dist_model(x_dist)
                backward_context = dummy_context
                if use_nosync and micro_step < micro_batch_steps-1:
                    backward_context = dist_optim.no_sync
                with backward_context():
                    y_dist.backward(dy)

                # Check that data tensors match
                torch.testing.assert_close(
                    y_dist, y_ref, rtol=rtol, atol=atol)
                torch.testing.assert_close(
                    x_dist.grad, x_ref.grad, rtol=rtol, atol=atol)

            # Optimization step
            ref_optim.step()
            dist_optim.step()

            # Check that parameters match
            for ref_param, dist_param in zip(ref_model.parameters(),
                                             dist_model.parameters()):
                torch.testing.assert_close(
                    dist_param, ref_param, rtol=rtol, atol=atol)

    def test_matches_pytorch_l2_reg(self):
        self.test_matches_pytorch(adam_w_mode=False)

    def test_matches_pytorch_no_overlap(self):
        self.test_matches_pytorch(
            overlap_communication=False,
            use_nosync=False,
        )

    def test_matches_pytorch_sync_every_step(self):
        self.test_matches_pytorch(use_nosync=False)

    def test_matches_pytorch_contiguous_buffers(self):
        self.test_matches_pytorch(contiguous_buffers=True)

    def test_matches_pytorch_fp64(self):
        self.test_matches_pytorch(
            rtol=1.3e-6,
            atol=1e-5,
            model_dtype=torch.float64,
            optim_dtype=torch.float32,
        )

    def test_matches_pytorch_fp16(self):
        self.test_matches_pytorch(
            rtol=5e-3,
            atol=1e-5,
            micro_batch_steps=1,
            model_dtype=torch.float16,
            optim_dtype=torch.float16,
        )

    def test_matches_pytorch_bf16(self):
        self.test_matches_pytorch(
            rtol=5e-2,
            atol=1e-5,
            micro_batch_steps=1,
            model_dtype=torch.bfloat16,
            optim_dtype=torch.bfloat16,
        )

    def test_matches_pytorch_fp16_params(self):
        self.test_matches_pytorch(
            rtol=5e-3,
            atol=1e-5,
            micro_batch_steps=1,
            model_dtype=torch.float16,
            optim_dtype=torch.float32,
            param_sync_dtype=torch.float16,
            store_params=True,
        )

    def test_matches_pytorch_bf16_grads(self):
        self.test_matches_pytorch(
            rtol=5e-2,
            atol=1e-5,
            micro_batch_steps=1,
            model_dtype=torch.float32,
            optim_dtype=torch.float32,
            grad_sync_dtype=torch.bfloat16,
        )

    def test_matches_pytorch_bf16_param_remainders(self):
        self.test_matches_pytorch(
            rtol=5e-2,
            atol=1e-5,
            micro_batch_steps=1,
            model_dtype=torch.bfloat16,
            optim_dtype=torch.float32,
            param_sync_dtype=torch.bfloat16,
            store_params=False,
            store_param_remainders=True,
        )

    def test_matches_pytorch_multi_dtypes(self):
        def init_optim(optim: DistributedFusedAdam):
            params = list(optim.parameters())
            optim.init_params(params[0::3], grad_sync_dtype=torch.bfloat16)
            optim.init_params(params[1::3], param_sync_dtype=torch.bfloat16)
        self.test_matches_pytorch(
            rtol=5e-2,
            atol=1e-5,
            init_optim_func=init_optim,
        )

    def test_matches_pytorch_int64_param_sync(self):
        self.test_matches_pytorch(
            param_sync_dtype=torch.int64,
        )

    def test_matches_pytorch_int32_param_sync_contiguous_buffers(self):
        self.test_matches_pytorch(
            param_sync_dtype=torch.int32,
            contiguous_buffers=True,
        )

    def test_matches_pytorch_uint8_param_sync(self):
        self.test_matches_pytorch(
            rtol=0.5,
            atol=0.05,
            model_dtype=torch.float16,
            optim_dtype=torch.float16,
            micro_batch_steps=1,
            param_sync_dtype=torch.uint8,
        )

    def test_matches_pytorch_scaled_state(self):
        self.test_matches_pytorch(
            rtol=5e-2,
            atol=1e-5,
            micro_batch_steps=1,
            model_dtype=torch.bfloat16,
            optim_dtype=torch.float16,
            param_sync_dtype=torch.int,
            store_params=True,
            with_scaled_states=True,
        )
    
    def test_matches_pytorch_nccl_ub(self):
        self.test_matches_pytorch(
            contiguous_buffers=True,
            nccl_ub=True,
        )

    def test_raises_on_mismatch(self):

        torch.manual_seed(self.seed + self.rank)

        # Identical models with data-parallel and ZeRO
        num_layers = 11
        layer_size = 7
        ref_model, ref_optim, dist_model, dist_optim = make_models(
            num_layers,
            layer_size,
        )

        # Only perform training step with distributed model
        dist_optim.zero_grad()
        x = torch.rand(3, layer_size) - 0.5
        x = x.to(dtype=torch.float32, device='cuda')
        dy = torch.rand_like(x) - 0.5
        y = dist_model(x)
        y.backward(dy)
        dist_optim.step()

        # Check that parameters do not match
        for ref_param, dist_param in zip(ref_model.parameters(),
                                         dist_model.parameters()):
            self.assertRaises(
                AssertionError,
                torch.testing.assert_close,
                dist_param, ref_param,
            )

    def test_clip_grad_norm(self):

        torch.manual_seed(self.seed + self.rank)

        # Identical models with data-parallel and ZeRO
        ref_model, ref_optim, dist_model, dist_optim = make_models(1, 1)

        # Training steps with pre-determined gradients
        xs = [3, 1, 4, 1, 5, 9]
        dys = [1, -1, 1, -1, 1, -1]
        for x, dy in zip(xs, dys):
            x = torch.tensor([[x]], dtype=torch.float32, device='cuda')
            dy = torch.tensor([[dy]], dtype=torch.float32, device='cuda')

            # Reference implementation
            ref_optim.zero_grad()
            y_ref = ref_model(x.detach())
            y_ref.backward(dy.detach())
            ref_grad_norm = torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 3.5)
            ref_optim.step()

            # Distributed implementation
            dist_optim.zero_grad()
            y_dist = dist_model(x.detach())
            y_dist.backward(dy.detach())
            dist_grad_norm = dist_optim.clip_grad_norm(3.5)
            dist_optim.step()

            # Check that parameters match
            torch.testing.assert_close(dist_grad_norm, ref_grad_norm)
            for ref_param, dist_param in zip(ref_model.parameters(),
                                             dist_model.parameters()):
                torch.testing.assert_close(dist_param, ref_param)

    def test_grad_scaler(self):

        torch.manual_seed(self.seed + self.rank)

        # Identical models with data-parallel and ZeRO
        ref_model, ref_optim, dist_model, dist_optim = make_models(1, 1)
        grad_scaler_args = dict(
            init_scale=3.21,
            growth_factor=1.23,
            backoff_factor=0.876,
            growth_interval=1,
        )
        ref_scaler =  torch.cuda.amp.GradScaler(**grad_scaler_args)
        dist_scaler =  torch.cuda.amp.GradScaler(**grad_scaler_args)

        # Training steps with pre-determined gradients
        xs = [3, 1, 4, 1, 5, 9]
        dys = [1, float('inf'), 1, 1, float('nan'), -1]
        for x, dy in zip(xs, dys):
            x = torch.tensor([[x]], dtype=torch.float32, device='cuda')
            dy = torch.tensor([[dy]], dtype=torch.float32, device='cuda')

            # Reference implementation
            ref_optim.zero_grad()
            y_ref = ref_model(x.detach())
            ref_scaler.scale(y_ref).backward(dy.detach())
            ref_scaler.step(ref_optim)
            ref_scaler.update()

            # Distributed implementation
            dist_optim.zero_grad()
            y_dist = dist_model(x.detach())
            dist_scaler.scale(y_dist).backward(dy.detach())
            dist_scaler.step(dist_optim)
            dist_scaler.update()

            # Check that parameters match
            for ref_param, dist_param in zip(ref_model.parameters(),
                                             dist_model.parameters()):
                torch.testing.assert_close(dist_param, ref_param)

    def test_checkpoint(
            self,
            rtol: Optional[float] = None,
            atol: Optional[float] = None,
            num_layers: int = 2,
            layer_size: int = 2,
            num_steps: int = 3,
            save_group_size: Optional[int] = None,
            load_group_size: Optional[int] = None,
            save_model_kwargs: Optional[dict] = None,
            load_model_kwargs: Optional[dict] = None,
    ):
        """Test state_dict and load_state_dict functions

        Two models are constructed, possibly on different process
        groups. One of the models is trained for a few steps, a
        checkpoint is saved, and the checkpoint is loaded on the other
        model. Both models are then trained for a few steps and
        checked to make sure that they produce identical results.

        Arguments:
            rtol (float): Relative tolerance for numerical checks (see
                torch.allclose).
            atol (float): Absolute tolerance for numerical checks (see
                torch.allclose).
            num_layers (int): Number of layers in test model.
            layer_size (int): Number of features in model layers.
            num_steps (int): Number of training steps to perform
                before and after checkpointing.
            save_group_size (int): Process group size for model that
                saves the checkpoint. Uses the default process group
                by default.
            load_group_size (int): Process group size for model that
                loads the checkpoint. Uses the default process group
                by default.
            save_model_kwargs (dict): keyword arguments passed to
                make_models when constructing the model that saves the
                checkpoint.
            load_model_kwargs (dict): keyword arguments passed to
                make_models when constructing the model that loads the
                checkpoint.

        """

        # Initialize process groups
        world_size = torch.distributed.get_world_size()
        if save_group_size is None:
            save_group_size = world_size
            save_group = None
        else:
            if save_group_size > world_size:
                self.skipTest(
                    f"Requires {save_group_size} ranks, found {world_size}"
                )
            save_ranks = list(range(save_group_size))
            save_group = torch.distributed.new_group(ranks=save_ranks)
        if load_group_size is None:
            load_group_size = world_size
            load_group = None
        else:
            if load_group_size > world_size:
                self.skipTest(
                    f"Requires {load_group_size} ranks, found {world_size}"
                )
            load_ranks = list(range(load_group_size))
            load_group = torch.distributed.new_group(ranks=load_ranks)

        # Construct two models with same config and different params
        torch.manual_seed(self.seed)
        if self.rank < save_group_size:
            if not save_model_kwargs:
                save_model_kwargs = {}
            _, _, model_save, optim_save = make_models(
                num_layers,
                layer_size,
                lr=0.1,
                process_group=save_group,
                average_grad_sync=False,
                overlap_communication=False,
                **save_model_kwargs,
            )
            optim_save.init_params(reversed(list(model_save.parameters())))
        torch.manual_seed(self.seed+1)
        if self.rank < load_group_size:
            if not load_model_kwargs:
                load_model_kwargs = {}
            _, _, model_load, optim_load = make_models(
                num_layers,
                layer_size,
                lr=1234.,
                process_group=load_group,
                average_grad_sync=False,
                overlap_communication=False,
                **load_model_kwargs,
            )
            optim_load.init_params(list(model_load.parameters()))

        batch_size = 2 * save_group_size * load_group_size
        def make_global_batch() -> torch.Tensor:
            """Generate random tensor on root rank and broadcast"""
            x = torch.empty(batch_size, layer_size, device='cuda')
            if self.rank == 0:
                torch.rand(x.size(), out=x)
                x -= 0.5
            torch.distributed.broadcast(x, src=0)
            return x

        def to_local_batch(
                global_batch: torch.Tensor,
                group: Optional[torch.distributed.ProcessGroup],
        ) -> Optional[torch.Tensor]:
            """Get local portion of tensor that is replicated across all ranks"""
            group_size = torch.distributed.get_world_size(group)
            if group_size < 0:
                return None
            local_batch_size = batch_size // group_size
            batch_start = self.rank * local_batch_size
            batch_end = (self.rank + 1) * local_batch_size
            return global_batch[batch_start:batch_end, ...]

        def to_global_batch(
                local_batch: torch.Tensor,
                group: Optional[torch.distributed.ProcessGroup],
        ) -> torch.Tensor:
            """Gather distributed tensor and broadcast to all ranks"""

            # Allocate buffer
            global_batch = torch.empty(batch_size, layer_size, device='cuda')

            # Gather data on root rank
            group_size = torch.distributed.get_world_size(group)
            if group_size > 0:
                local_batches = None
                if self.rank == 0:
                    local_batch_size = batch_size // group_size
                    local_batches = [
                        global_batch[rank*local_batch_size:(rank+1)*local_batch_size, ...]
                        for rank in range(group_size)
                    ]
                torch.distributed.gather(
                    local_batch,
                    local_batches,
                    dst=0,
                    group=group,
                )

            # Broadcast data to all ranks
            torch.distributed.broadcast(global_batch, src=0)
            return global_batch

        # Train one of the models
        torch.manual_seed(self.seed+2)
        for step in range(num_steps):
            if self.rank < save_group_size:
                optim_save.zero_grad()
            x = make_global_batch()
            dy = make_global_batch()
            if self.rank < save_group_size:
                x = to_local_batch(x, save_group)
                dy = to_local_batch(dy, save_group)
                y = model_save(x)
                y.backward(dy)
                optim_save.step()

        # Make sure models are different
        if self.rank < min(save_group_size, load_group_size):
            for param_save, param_load in zip(model_save.parameters(),
                                              model_load.parameters()):
                self.assertRaises(
                    AssertionError,
                    torch.testing.assert_close,
                    param_load,
                    param_save,
                    rtol=rtol,
                    atol=atol,
                )

        # Save state
        state_bytes = None
        if self.rank < save_group_size:
            state_dict = {
                'model': model_save.state_dict(),
                'optim': optim_save.state_dict(),
            }
            byte_stream = io.BytesIO()
            torch.save(state_dict, byte_stream)
            state_bytes = byte_stream.getvalue()

        # Broadcast state from root rank and load
        if self.rank < load_group_size:
            if load_group_size != save_group_size:
                if self.rank != 0:
                    state_bytes = None
                state_bytes = [state_bytes]
                torch.distributed.broadcast_object_list(
                    state_bytes,
                    src=0,
                    group=load_group,
                )
                state_bytes = state_bytes[0]
            state_dict = torch.load(io.BytesIO(state_bytes))
            model_load.load_state_dict(state_dict['model'])
            optim_load.load_state_dict(state_dict['optim'])

        # Make sure models are identical
        if self.rank < min(save_group_size, load_group_size):
            for param_save, param_load in zip(model_save.parameters(),
                                              model_load.parameters()):
                torch.testing.assert_close(
                    param_load,
                    param_save,
                    rtol=rtol,
                    atol=atol
                )

        # Train both models
        torch.manual_seed(self.seed+3)
        for step in range(num_steps):

            # Reset grads
            if self.rank < save_group_size:
                optim_save.zero_grad()
            if self.rank < load_group_size:
                optim_load.zero_grad()

            # Synthetic data
            x = make_global_batch()
            dy = make_global_batch()

            # Training step for model that saved checkpoint
            y_save = None
            dx_save = None
            if self.rank < save_group_size:
                x_save = to_local_batch(x, save_group)
                x_save = x_save.detach().clone().requires_grad_(True)
                dy_save = to_local_batch(dy, save_group)
                y_save = model_save(x_save)
                y_save.backward(dy_save)
                dx_save = x_save.grad
            y_save = to_global_batch(y_save, save_group)
            dx_save = to_global_batch(dx_save, save_group)

            # Training step for model that loaded checkpoint
            y_load = None
            dx_load = None
            if self.rank < load_group_size:
                x_load = to_local_batch(x, load_group)
                x_load = x_load.detach().clone().requires_grad_(True)
                dy_load = to_local_batch(dy, load_group)
                y_load = model_load(x_load)
                y_load.backward(dy_load)
                dx_load = x_load.grad
            y_load = to_global_batch(y_load, load_group)
            dx_load = to_global_batch(dx_load, load_group)

            # Check that data tensors match
            torch.testing.assert_close(y_load, y_save, rtol=rtol, atol=atol)
            torch.testing.assert_close(dx_load, dx_save, rtol=rtol, atol=atol)

            # Optimizer step
            if self.rank < save_group_size:
                optim_save.step()
            if self.rank < load_group_size:
                optim_load.step()

            # Check that parameters match
            if self.rank < min(save_group_size, load_group_size):
                for param_save, param_load in zip(model_save.parameters(),
                                                  model_load.parameters()):
                    torch.testing.assert_close(
                        param_load,
                        param_save,
                        rtol=rtol,
                        atol=atol,
                    )

    def test_checkpoint_save_1gpu(self):
        """Test loading checkpoint with one GPU"""
        self.test_checkpoint(save_group_size=1)

    def test_checkpoint_load_1gpu(self):
        """Test saving checkpoint with one GPU"""
        self.test_checkpoint(load_group_size=1)

    def test_checkpoint_bf16(self):
        """Test checkpoint with BF16 model"""
        self.test_checkpoint(
            rtol=5e-2,
            atol=1e-5,
            save_model_kwargs=dict(
                model_dtype=torch.bfloat16,
                optim_dtype=torch.float32,
                param_sync_dtype=torch.bfloat16,
                store_params=False,
                store_param_remainders=True,
            ),
            load_model_kwargs=dict(
                model_dtype=torch.bfloat16,
                optim_dtype=torch.float32,
                param_sync_dtype=torch.bfloat16,
                store_params=False,
                store_param_remainders=True,
            ),
        )

    def test_checkpoint_scaled_state(self):
        """Test checkpoint with scaled FP16 state"""
        self.test_checkpoint(
            rtol=5e-2,
            atol=1e-5,
            save_model_kwargs=dict(
                model_dtype=torch.bfloat16,
                optim_dtype=torch.float16,
                param_sync_dtype=torch.int,
                store_params=True,
                with_scaled_states=True,
            ),
            load_model_kwargs=dict(
                model_dtype=torch.bfloat16,
                optim_dtype=torch.float16,
                param_sync_dtype=torch.int,
                store_params=True,
                with_scaled_states=True,
            ),
        )

    def test_bucket_low_utilization_warning(self):
        """Test warning when bucket utilization is low"""
        layer_size = 2*1024*1024
        num_layers = 4
        fairish_bucket_cap_mb = 4*num_layers*layer_size/(1024*1024)

        # Check that warning is raised when bucket utilization is low
        with self.assertWarnsRegex(Warning, ".*Consider decreasing the bucket_cap_mb argument."):
            self.test_matches_pytorch(
                num_layers=num_layers,
                layer_size=layer_size,
                overlap_communication=False,
                bucket_cap_mb=fairish_bucket_cap_mb * 2,
            )

        # Check that warning is not raised when bucket utilization is high
        with warnings.catch_warnings(record=True) as warns:
            self.test_matches_pytorch(
                num_layers=num_layers,
                layer_size=layer_size,
                overlap_communication=False,
                bucket_cap_mb=fairish_bucket_cap_mb,
            )
            for w in warns:
                self.assertNotRegex(str(w.message), ".*Consider decreasing the bucket_cap_mb argument.")


if __name__ == "__main__":
    # Assume script has been run with torchrun
    common_utils.run_tests()
