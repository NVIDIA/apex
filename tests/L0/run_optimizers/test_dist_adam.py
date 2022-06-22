import argparse
import os
import random

import torch
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam

class TestModel(torch.nn.Module):

    def __init__(self, args):
        super(TestModel, self).__init__()
        self.linear = torch.nn.Sequential(*[
            torch.nn.Linear(args.dim, args.dim)
            for _ in range(args.layers)
        ])

    def forward(self, x):
        y = 0
        for i, l in enumerate(self.linear):
            y += (i+1) * l(x)
        return y

def setup(args):

    # Construct models with same parameters
    ref_model = TestModel(args).float().cuda()
    dist_model = TestModel(args).float().cuda()
    with torch.no_grad():
        for ref_param, dist_param in zip(dist_model.parameters(),
                                         ref_model.parameters()):
            dist_param.data.copy_(ref_param.data)
    ref_model = torch.nn.parallel.DistributedDataParallel(
        ref_model,
        device_ids=[args.rank],
        output_device=args.rank,
    )

    # Construct optimizers with same hyperparameters
    optim_args = { 'lr': 1, 'betas': (0.5,0.75), 'eps': 0.1, 'weight_decay': 0.1 }
    ref_optim = torch.optim.AdamW(
        [
            {'params': list(ref_model.parameters())[1::2], 'lr': 0.5},
            {'params': list(ref_model.parameters())[0::2]},
        ],
        **optim_args,
    )
    dist_optim = DistributedFusedAdam(
        [
            {'params': list(dist_model.parameters())[1::2], 'lr': 0.5},
            {'params': list(dist_model.parameters())[0::2]},
        ],
        bucket_cap_mb=71/(4*1024*1024),
        **optim_args,
    )

    return ref_model, ref_optim, dist_model, dist_optim

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--dim', type=int, default=7)
    parser.add_argument('--layers', type=int, default=11)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)

    args = parser.parse_args()

    return args

def setup_env(args):

    # Initialize NCCL
    local_rank = args.local_rank
    if local_rank < 0:
        local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()

    # Initialize RNG
    seed = 42 + args.rank
    random.seed(seed)
    torch.manual_seed(seed)

    return args

def main():
    args = parse_args()
    args = setup_env(args)
    torch.set_printoptions(precision=16)

    def assert_allclose(ref_x, dist_x, message):
        message = (
            f'Rank {args.rank}: {message}\n'
            f'Reference Adam: {ref_x}\n'
            f'Distributed Adam: {dist_x}\n'
            f'Relative error: {torch.abs((ref_x-dist_x)/ref_x)}\n'
        )
        assert torch.allclose(ref_x, dist_x, atol=args.atol, rtol=args.rtol), message

    # Train model with data-parallelism and ZeRO
    ref_model, ref_optim, dist_model, dist_optim = setup(args)
    for step in range(args.steps):

        # Synthetic data
        x = torch.randn(args.batch, args.dim).cuda()
        dy = torch.randn_like(x).cuda()

        # Reference implementation
        ref_optim.zero_grad()
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = ref_model(x_ref)
        y_ref.backward(dy)
        ref_optim.step()

        # Distributed implementation
        dist_optim.zero_grad()
        x_dist = x.detach().clone().requires_grad_(True)
        y_dist = dist_model(x_dist)
        y_dist.backward(dy)
        dist_optim.step()

        # Check values
        torch.cuda.synchronize()
        torch.distributed.barrier()
        assert_allclose(
            y_ref,
            y_dist,
            f'inconsistent output in step {step}',
        )
        assert_allclose(
            x_ref.grad,
            x_dist.grad,
            f'inconsistent input grad in step {step}',
        )
        for i, (ref_param, dist_param) in enumerate(zip(ref_model.parameters(),
                                                        dist_model.parameters())):
            assert_allclose(
                ref_param,
                dist_param,
                f'inconsistent param {i} in step {step}',
            )

if __name__ == "__main__":
    main()
