import copy
import typing
import unittest

import torch
import torch.nn as nn
from torch.testing._internal import common_utils

SKIP_TEST = None
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
try:
    from apex.contrib.cudnn_gbn import GroupBatchNorm2d as GBN
except ImportError as e:
    SKIP_TEST = e


# Usage: python /path/to/cudnn_gbn/test_gbn_with_two_gpus.py

input_shapes = [
    [1, 1024, 48, 72],
    [1, 128, 192, 288],
    [1, 128, 384, 576],
    [1, 1536, 48, 72],
    [1, 2048, 48, 72],
    [1, 256, 1, 1],
    [1, 256, 192, 288],
    [1, 256, 384, 576],
    [1, 256, 48, 72],
    [1, 256, 96, 144],
    [1, 32, 384, 576],
    [1, 48, 192, 288],
    [1, 64, 384, 576],
    [1, 728, 48, 72],
    [1, 728, 96, 144],
]


class BNModelRef(nn.Module):
    def __init__(self, num_features, num_layers=1000):
        super().__init__()
        self.fwd = nn.Sequential(
            *[
                nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        return self.fwd(x)


class BNModel(nn.Module):
    def __init__(self, num_features, num_layers=1000):
        super().__init__()
        self.fwd = nn.Sequential(
            *[
                GBN(num_features, group_size=2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        return self.fwd(x)


def get_rand_tensors(global_shape, device):
    inp_t = torch.rand(global_shape, dtype=torch.float32, device=device).to(memory_format=torch.channels_last)
    weight = torch.rand(global_shape[1], dtype=torch.float32, device=device)
    bias = torch.rand(global_shape[1], dtype=torch.float32, device=device)
    _grad_out = torch.rand(global_shape, dtype=torch.float32, device=device).to(memory_format=torch.channels_last)
    return inp_t, weight, bias, _grad_out


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class TestCudnnGBN(NcclDistributedTestBase):
    def _prep(self):
        torch.cuda.manual_seed(333)
        torch.manual_seed(333)

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @torch.backends.cudnn.flags(enabled=True, benchmark=True)
    def _test_cudnn_gbn(
        self,
        num_layers: int,
        shape: typing.List[int],
        *,
        memory_format: torch.memory_format = torch.channels_last,
    ) -> None:
        global_shape = copy.deepcopy(shape)
        global_shape[0] = self.world_size

        device = torch.device("cuda", self.rank)
        cudnn_gbn_model = BNModel(
            num_features=shape[1],
            num_layers=num_layers,
        ).to(device=device, memory_format=memory_format)
        ref_model = BNModelRef(
            num_features=shape[1],
            num_layers=num_layers,
        ).to(device=device, memory_format=memory_format)

        input, weight, bias, grad_out = get_rand_tensors(global_shape, device)
        with torch.no_grad():
            ref_model.fwd[0].weight.copy_(weight)
            ref_model.fwd[0].bias.copy_(bias)
            cudnn_gbn_model.fwd[0].weight.copy_(weight)
            cudnn_gbn_model.fwd[0].bias.copy_(bias)

            ref_input = input.clone().detach().requires_grad_()
            input = input[self.rank : self.rank + 1, ...].clone().detach().requires_grad_()

            ref_grad_out = grad_out.half().clone().detach()
            grad_out = grad_out[self.rank : self.rank + 1, ...].half().clone().detach()

        with torch.cuda.amp.autocast():
            out = cudnn_gbn_model(input)
            ref_out = ref_model(ref_input.half())
        out.backward(grad_out)
        ref_out.backward(ref_grad_out)

        kwargs = {"rtol": 3.5e-3, "atol": 3e-2, "msg": f"shape: {shape}"}

        torch.testing.assert_close(ref_out[self.rank : self.rank + 1], out, **kwargs)
        torch.testing.assert_close(ref_input.grad[self.rank : self.rank + 1], input.grad, **kwargs)
        # compensating the averaging over processes done by DDP
        # in order to produce mathematically equivalent result
        # https://github.com/NVIDIA/apex/issues/134#issuecomment-458307368
        torch.testing.assert_close(
            ref_model.fwd[0].weight.grad / self.world_size, cudnn_gbn_model.fwd[0].weight.grad, **kwargs
        )
        torch.testing.assert_close(
            ref_model.fwd[0].bias.grad / self.world_size, cudnn_gbn_model.fwd[0].bias.grad, **kwargs
        )

    def test_cudnngbn(self):
        if self.world_size != 2:
            self.skipTest(f"This test is written for world_size of 2 but {self.world_size}")
        for shape in input_shapes:
            self._prep()
            self._test_cudnn_gbn(1, shape)


if __name__ == "__main__":
    common_utils.run_tests()
