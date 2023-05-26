import os
import inspect
import torch
from torch.cuda.amp import GradScaler
from torch.testing._internal import common_utils
from apex.parallel.distributed import flat_dist_call
from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase

def get_init_weights_func():
    @torch.no_grad()
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            m.weight.fill_(1.0)
    return init_weights

class ModelFoo(torch.nn.Module):
    def __init__(self):
        super(ModelFoo, self).__init__()
        self.linear = torch.nn.Linear(128, 128, bias = False)
        self.loss = torch.nn.MSELoss()

    def forward(self, input_tensor, gt):
        y = self.linear(input_tensor)
        loss = self.loss(y, gt)
        return loss

# A test for distributed fused Lamb optimizer: run several iterations and see if loss decreases
# There are two instances of the same test because based on `world_size` the optimizer decides what collectives operation to use. 
# If torch.distributed.get_world_size() == torch.cuda.device_count() it uses only `all_gather`.
# If torch.distributed.get_world_size() < torch.cuda.device_count() it uses both `all_gather` and `reduce_scatter`.
class NcclDistributedFusedLAMB(NcclDistributedTestBase):
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @common_utils.parametrize("no_copy", [False, True])
    @common_utils.parametrize("opt_kwargs", [
        dict(overlap_reductions=True, dwu_num_blocks=2, dwu_num_chunks=2,
             fused_norm=False, fuse_scale=False, clip_after_ar=True,
             full_ar=False),
        dict(overlap_reductions=False, dwu_num_blocks=1, dwu_num_chunks=1,
             fused_norm=True, fuse_scale=True, clip_after_ar=False),
    ])
    def test_distributed_fused_lamb(self, no_copy, opt_kwargs):
        if no_copy and 'no_copy' not in inspect.getfullargspec(torch.distributed.reduce_scatter).args:
            self.skipTest("does not support no_copy")
        if no_copy and 'no_copy' not in inspect.getfullargspec(torch.distributed.all_gather).args:
            self.skipTest("does not support no_copy")

        assert torch.distributed.is_initialized()
        gpu_count = torch.distributed.get_world_size()

        init_scale = 100
        lr = torch.tensor(0.1).cuda()
        grad_scaler = GradScaler(init_scale=init_scale, growth_interval=1000)

        model = ModelFoo()
        model = model.cuda().half()
        model.apply(get_init_weights_func())

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if 'full_ar' not in opt_kwargs:
            opt_kwargs['full_ar'] = gpu_count == torch.cuda.device_count()

        # Aidyn-A: not sure what parameters are the best for testing purposes, 
        # setting up whatever I think appropriate. 
        optimizer = DistributedFusedLAMB(
                optimizer_grouped_parameters, 
                lr=0.1,
                betas=(0.9, 0.9),
                eps=1e-6,
                max_grad_norm=1.0,
                dwu_group_size=gpu_count,
                dwu_num_rs_pg=1,
                dwu_num_ar_pg=1,
                dwu_num_ag_pg=1,
                use_nvlamb=False,
                set_param_views_to_flat_buffer=False,
                e5m2_allgather=False,
                **opt_kwargs
        )
        optimizer.set_global_scale(init_scale)

        optimizer._reduce_scatter_no_copy = no_copy
        optimizer._all_gather_no_copy = no_copy

        flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,) )

        x = torch.randn(4096, 128, dtype=torch.float16).cuda()
        y = torch.randn(4096, 128, dtype=torch.float16).cuda()

        losses = []
        for _ in range(10):
            loss = model(x, y)
            optimizer._lazy_init_stage1()
            grad_scaler.scale(loss).backward()
            optimizer._lazy_init_stage2()
            optimizer._lr = lr
            optimizer.complete_reductions()
            optimizer.set_global_scale(grad_scaler._get_scale_async())
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        self.assertTrue(losses == sorted(losses, reverse=True))

common_utils.instantiate_parametrized_tests(NcclDistributedFusedLAMB)

class NcclDistributedFusedLAMB_partial_ar(NcclDistributedFusedLAMB):
    @property
    def world_size(self) -> int:
        return max(torch.cuda.device_count()-1, 1)

if __name__ == "__main__":
    common_utils.run_tests()

