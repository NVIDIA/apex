import argparse
import random
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from apex import amp
from apex.optimizers import FusedAdam
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam


class TestModel(torch.nn.Module):
    def __init__(self, args):
        super(TestModel, self).__init__()

        self.linear = torch.nn.Sequential(*[torch.nn.Linear(args.dim, args.dim, bias=args.bias) for _ in range(args.layers)])

    def forward(self, x):
        return self.linear(x)

def setup(args):
    ## Model
    ref_model = TestModel(args).cuda()
    dist_model = TestModel(args).cuda()

    # Same weights
    with torch.no_grad():
        for dp, rp in zip(dist_model.parameters(), ref_model.parameters()):
            dp.data.copy_(rp.data)

    dist_model = dist_model.half()


    ## Optimizer
    # same hyperparameters
    ref_opt_args = { 'lr': 1e-3, 'eps': 1e-6, 'weight_decay': 0.01 }
    ref_opt = FusedAdam(ref_model.parameters(), **ref_opt_args)

    dist_opt_args = ref_opt_args.copy()
    dist_opt_args.update( {'overlap_reductions' : False} )
    dist_opt_args.update( {'process_group_size' : args.n_gpu} )
    dist_opt_args.update( {'dwu_group_size' : args.dwu_group_size} )
    dist_opt_args.update( {'dwu_num_blocks' : 1} )
    dist_opt_args.update( {'dwu_num_chunks' : 1} )
    dist_opt = DistributedFusedAdam(dist_model.parameters(), **dist_opt_args)
    dist_opt.set_global_scale(1.)
    
    ## amp-init
    amp_args = { 'loss_scale' : 'dynamic' , 'opt_level' : 'O2'}
    ref_model, ref_opt = amp.initialize(ref_model, ref_opt, **amp_args)
    
   
    ## DDP
    ref_model = DDP(ref_model, device_ids=[args.rank])
    with torch.no_grad():
        for dp in dist_model.parameters():
             torch.distributed.broadcast(dp.data, src=0)
        for rp in ref_model.parameters():
            torch.distributed.broadcast(rp.data, src=0)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    if get_rank() == 0:
        print(f'dist opt with {args.n_gpu} GPUs')

    return ref_model, ref_opt, dist_model, dist_opt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1)
    parser.add_argument('--dwu_group_size', type=float, default=1)

    args = parser.parse_args()

    return args

def setup_env(args):
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.rank = torch.distributed.get_rank()
    args.n_gpu = torch.distributed.get_world_size()

    seed = 42 + get_rank()

    random.seed(seed)
    torch.manual_seed(seed)

    return args

def get_rank():
    return torch.distributed.get_rank()

def main():
    args = parse_args()
    args = setup_env(args)
    tol_args = { 'atol' : args.atol, 'rtol' : args.rtol }

    torch.set_printoptions(precision=16)

    ref_model, ref_opt, dist_model, dist_opt = setup(args)

    # lazy_init not called yet, initialize stash
    stash = ref_opt._amp_stash
    stash.all_fp16_params, stash.all_fp32_from_fp16_params = [], []

    # make sure everything from _first_step_init_ is ready before training
    # e.g. registering allreduce_hook
    # so that gradients are copied/reduced when necessary
    dist_opt._init_everything()

    for i in range(args.steps):
        x_ref = torch.randn(args.batch, args.dim, dtype=torch.half).cuda().requires_grad_(True)
        x_dist = x_ref.clone().detach().requires_grad_(True)
        
        if get_rank() == 0:
            print(f'[{i}] Checking input')
        #print("x_ref:", x_ref.flatten()[:10])
        #print("x_dist:", x_dist.flatten()[:10])
        assert(torch.allclose(x_ref, x_dist, **tol_args))

        y_ref = ref_model(x_ref).half()
        y_dist = dist_model(x_dist)

        if get_rank() == 0:
            print(f'[{i}] Checking output')
        #print("y_ref:", y_ref.flatten()[:10])
        #print("y_dist:", y_dist.flatten()[:10])
        assert(torch.allclose(y_ref, y_dist, **tol_args))

        dy = torch.randn_like(y_ref)

        y_ref.backward(dy)
        y_dist.backward(dy)

        if get_rank() == 0:
            print(f'[{i}] Checking gradients')
        torch.distributed.barrier()
        torch.cuda.synchronize()
        assert(torch.allclose(x_ref.grad, x_dist.grad, **tol_args))

        # gradient all-reduce within distributed optimizer
        dist_opt.complete_reductions()

        if get_rank() == 0:
            print(f'[{i}] Stepping')
        ref_opt.step()
        dist_opt.step()

        torch.cuda.synchronize()
        torch.distributed.barrier()
        print('Checking new weights')
        if get_rank() == 0:
            print("ref param:", ref_model.module.linear[0].weight)
            print("dist param:", dist_model.linear[0].weight)
        
        for i, (rp, dp) in enumerate(zip(ref_model.parameters(), dist_model.parameters())):
            if not torch.allclose(rp, dp, **tol_args):
                if get_rank() == 0:
                    print(f'Rank: {get_rank()}, Param: {i}')
                    print(f'ref: {rp.sum().item()}, dist: {dp.sum().item()}')
                    print(rp)
                    print(dp)
    
                    print(torch.abs(rp-dp) > tol_args['atol'])
                    sys.exit(0)

        # zero grads
        for rp, dp in zip(ref_model.parameters(), dist_model.parameters()):
            rp.grad = None
            dp.grad = None


if __name__ == "__main__":
    main()

