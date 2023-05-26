from contextlib import contextmanager
import io
import unittest

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
        num_layers,
        size,
        adam_w_mode=True,
        model_dtype=torch.float32,
        optim_dtype=None,
        grad_sync_dtype=None,
        param_sync_dtype=None,
        device='cuda',
        overlap_communication=True,
        contiguous_buffers=False,
        store_params=False,
        store_param_remainders=False,
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
    )

    # Construct optimizers with same hyperparameters
    if optim_dtype is None:
        optim_dtype = model_dtype
    optim_args = dict(lr=0.1, betas=(0.1,0.2), eps=0.25, weight_decay=0.1)
    ref_optim_class = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    ref_optim = ref_optim_class(
        [
            {'params': list(ref_model.parameters())[1::2], 'lr': 0.2},
            {'params': list(ref_model.parameters())[0::2]},
        ],
        **optim_args,
    )
    dist_optim = DistributedFusedAdam(
        [
            {'params': list(dist_model.parameters())[1::2], 'lr': 0.2},
            {'params': list(dist_model.parameters())[0::2]},
        ],
        adam_w_mode=adam_w_mode,
        overlap_grad_sync=overlap_communication,
        overlap_param_sync=overlap_communication,
        bucket_cap_mb=71/(4*1024*1024),
        dtype=optim_dtype,
        grad_sync_dtype=grad_sync_dtype,
        param_sync_dtype=param_sync_dtype,
        contiguous_param_buffer=contiguous_buffers,
        contiguous_grad_buffer=contiguous_buffers,
        store_params=store_params,
        store_param_remainders=store_param_remainders,
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
            rtol=None,
            atol=None,
            num_layers=11,
            layer_size=7,
            batch_size=3,
            num_steps=3,
            micro_batch_steps=3,
            adam_w_mode=True,
            overlap_communication=True,
            use_nosync=True,
            model_dtype=torch.float32,
            optim_dtype=None,
            grad_sync_dtype=None,
            param_sync_dtype=None,
            device='cuda',
            contiguous_buffers=False,
            store_params=False,
            store_param_remainders=False,
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
            contiguous_buffers=contiguous_buffers,
            store_params=store_params,
            store_param_remainders=store_param_remainders,
        )

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

    def test_checkpoint(self):

        # Construct two models with same config and different params
        num_layers = 5
        layer_size = 2
        torch.manual_seed(self.seed + self.rank)
        _, _, model_save, optim_save = make_models(num_layers, layer_size)
        _, _, model_load, optim_load = make_models(num_layers, layer_size)

        # Train one of the models
        num_steps = 3
        micro_batch_steps = 2
        batch_size = 4
        for step in range(num_steps):
            optim_save.zero_grad()
            for micro_step in range(micro_batch_steps):
                x = torch.rand(batch_size, layer_size) - 0.5
                dy = torch.rand_like(x) - 0.5
                x = x.cuda()
                dy = dy.cuda()
                y = model_save(x)
                y.backward(dy)
            optim_save.step()

        # Make sure models are different
        for param_save, param_load in zip(model_save.parameters(),
                                          model_load.parameters()):
            self.assertRaises(
                AssertionError,
                torch.testing.assert_close,
                param_load, param_save,
            )

        # Save state on root rank and load on all ranks
        state_dict = {
            'model': model_save.state_dict(),
            'optim': optim_save.state_dict(),
        }
        if self.rank == 0:
            state_bytes = io.BytesIO()
            torch.save(state_dict, state_bytes)
            state_bytes = [state_bytes.getvalue()]
        else:
            state_bytes = [None]
        torch.distributed.broadcast_object_list(state_bytes, src=0)
        state_bytes = io.BytesIO(state_bytes[0])
        state_dict = torch.load(state_bytes)
        model_load.load_state_dict(state_dict['model'])
        optim_load.load_state_dict(state_dict['optim'])

        # Make sure models are identical
        for param_save, param_load in zip(model_save.parameters(),
                                          model_load.parameters()):
            torch.testing.assert_close(param_load, param_save)

        # Train both models
        num_steps = 3
        micro_batch_steps = 3
        batch_size = 5
        for step in range(num_steps):

            # Reset gradients
            optim_save.zero_grad()
            optim_load.zero_grad()

            # Forward and backward passes
            for micro_step in range(micro_batch_steps):

                # Synthetic data
                x = torch.rand(batch_size, layer_size) - 0.5
                dy = torch.rand_like(x) - 0.5
                x = x.cuda()
                dy = dy.cuda()

                # Forward and backward pass
                x_save = x.detach().clone().requires_grad_(True)
                y_save = model_save(x_save)
                y_save.backward(dy)
                x_load = x.detach().clone().requires_grad_(True)
                y_load = model_load(x_load)
                y_load.backward(dy)

                # Check that data tensors match
                torch.testing.assert_close(y_load, y_save)
                torch.testing.assert_close(x_load.grad, x_save.grad)

            # Optimizer step
            optim_save.step()
            optim_load.step()

            # Check that parameters match
            for param_save, param_load in zip(model_save.parameters(),
                                              model_load.parameters()):
                torch.testing.assert_close(param_load, param_save)

if __name__ == "__main__":
    # Assume script has been run with torchrun
    common_utils.run_tests()
