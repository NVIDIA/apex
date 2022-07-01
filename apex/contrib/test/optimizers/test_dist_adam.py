from contextlib import contextmanager
import os

import torch
from torch.testing._internal import common_utils
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase

class SimpleModel(torch.nn.Module):

    def __init__(self, num_layers, size):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(size, size, bias=(i%3==0))
            for i in range(num_layers)
        ])

    def forward(self, x):
        y = 0
        for i, l in enumerate(self.layers):
            y += (i+1) * l(x)
        return y

def make_models(
        num_layers,
        size,
        dtype=torch.float32,
        device='cuda',
        overlap_communication=True,
):

    # Construct models with same parameters
    ref_model = SimpleModel(num_layers, size).to(dtype=dtype, device=device)
    dist_model = SimpleModel(num_layers, size).to(dtype=dtype, device=device)
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
    optim_args = dict(lr=0.1, betas=(0.1,0.2), eps=0.25, weight_decay=0.1)
    ref_optim = torch.optim.AdamW(
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
        overlap_grad_sync=overlap_communication,
        bucket_cap_mb=71/(4*1024*1024),
        **optim_args,
    )

    return ref_model, ref_optim, dist_model, dist_optim

@contextmanager
def dummy_context():
    try:
        yield
    finally:
        pass

class TestDistributedFusedAdam(NcclDistributedTestBase):

    seed = 1234

    def test_matches_pytorch(
            self,
            num_layers=11,
            layer_size=7,
            batch_size=3,
            num_steps=3,
            micro_batch_steps=3,
            overlap_communication=True,
            use_nosync=True,
            dtype=torch.float32,
            device='cuda',
            rtol=None,
            atol=None,
    ):

        torch.manual_seed(self.seed + self.rank)

        # Identical models with data-parallel and ZeRO
        ref_model, ref_optim, dist_model, dist_optim = make_models(
            num_layers,
            layer_size,
            dtype=dtype,
            device=device,
            overlap_communication=overlap_communication,
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
                x = x.to(dtype=dtype, device=device)
                dy = dy.to(dtype=dtype, device=device)

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

    def test_matches_pytorch_no_overlap(self):
        self.test_matches_pytorch(
            overlap_communication=False,
            use_nosync=False,
        )

    def test_matches_pytorch_sync_every_step(self):
        self.test_matches_pytorch(use_nosync=False)

    def test_matches_pytorch_fp64(self):
        self.test_matches_pytorch(
            dtype=torch.float64,
            rtol=1.3e-6,
            atol=1e-5,
        )

    def test_matches_pytorch_fp16(self):
        self.test_matches_pytorch(
            dtype=torch.float16,
            rtol=1e-2,
            atol=1e-2,
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
        x = torch.rand(3, layer_size) + 0.5
        x = x.to(dtype=torch.float32, device='cuda')
        dy = torch.rand_like(x) + 0.5
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
            x = torch.tensor([x], dtype=torch.float32, device='cuda')
            dy = torch.tensor([dy], dtype=torch.float32, device='cuda')

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
            x = torch.tensor([x], dtype=torch.float32, device='cuda')
            dy = torch.tensor([dy], dtype=torch.float32, device='cuda')

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

if __name__ == "__main__":
    # Assume script has been run with torchrun
    common_utils.run_tests()
