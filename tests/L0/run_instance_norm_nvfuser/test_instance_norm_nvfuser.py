import itertools
import unittest

import torch
import torch.nn as nn

import apex
from apex.normalization import InstanceNorm3dNVFuser


class TestInstanceNormNVFuser(unittest.TestCase):
    dtype = torch.float
    track_running_stats = False
    channels_last = False
    affine = False
    batch_size = 5
    channel_size = 7
    spatial_size = 3

    def init_modules(self):
        self.m = InstanceNorm3dNVFuser(self.channel_size, affine=self.affine, track_running_stats=self.track_running_stats, device='cuda', dtype=self.dtype)
        self.reference_m = torch.nn.InstanceNorm3d(self.channel_size, affine=self.affine, track_running_stats=self.track_running_stats, device='cuda', dtype=self.dtype)

    def check_same_output(self):
        torch.manual_seed(42)
        for i in range(2): # exercise JIT + caching
            inp = torch.randint(0, 2, (self.batch_size, self.channel_size, self.spatial_size, self.spatial_size, self.spatial_size), device='cuda', requires_grad=True, dtype=self.dtype)
            inp2 = inp.detach().clone()
            inp2.requires_grad = True
            if self.channels_last:
                _inp = inp.to(memory_format=torch.channels_last_3d)
            else:
                _inp = inp
            out = self.m(_inp)
            out2 = self.reference_m(inp2)
            if self.m.running_mean is None:
                assert self.reference_m.running_mean is None
                assert self.m.running_var is None
                assert self.reference_m.running_var is None
            else:
                torch.testing.assert_close(self.m.running_mean, self.reference_m.running_mean)
                if self.dtype == torch.float16:
                    torch.testing.assert_close(self.m.running_var, self.reference_m.running_var, atol=5e-3, rtol=5e-3)
                else:
                    torch.testing.assert_close(self.m.running_var, self.reference_m.running_var)
            torch.testing.assert_close(out, out2)
            grad_out = torch.randn_like(inp)
            out.backward(grad_out)
            out2.backward(grad_out)
            if self.dtype == torch.float16:
                torch.testing.assert_close(inp.grad, inp2.grad, atol=5e-3, rtol=5e-3)
            elif self.dtype == torch.bfloat16:
                torch.testing.assert_close(inp.grad, inp2.grad, atol=2e-2, rtol=2e-2)
            else:
                torch.testing.assert_close(inp.grad, inp2.grad)
            if self.m.weight is not None:
                if self.dtype == torch.float16:
                    torch.testing.assert_close(self.m.weight.grad, self.reference_m.weight.grad, atol=5e-2, rtol=5e-2)
                elif self.dtype == torch.bfloat16:
                    torch.testing.assert_close(self.m.weight.grad, self.reference_m.weight.grad, atol=7e-2, rtol=8e-2)
                else:
                    torch.testing.assert_close(self.m.weight.grad, self.reference_m.weight.grad)
            if self.m.bias is not None:
                if self.dtype == torch.float16:
                    torch.testing.assert_close(self.m.bias.grad, self.reference_m.bias.grad, atol=5e-3, rtol=7e-2)
                elif self.dtype == torch.bfloat16:
                    torch.testing.assert_close(self.m.bias.grad, self.reference_m.bias.grad, atol=5e-2, rtol=1e-2)
                else:
                    torch.testing.assert_close(self.m.bias.grad, self.reference_m.bias.grad)

    def test_sweep(self):
        dtypes = [torch.float, torch.half]
        if torch.cuda.get_device_capability() >= (8, 0):
            dtypes.append(torch.bfloat16)
        for dtype, track_running_stats, channels_last, affine in itertools.product(dtypes, (False, True), (False, True), (False, True)):
            with self.subTest(dtype=dtype, track_running_stats=track_running_stats, channels_last=channels_last, affine=affine):
                self.dtype = dtype
                self.track_running_stats = track_running_stats
                self.channels_last = channels_last
                self.affine = affine
                self.init_modules()
                self.check_same_output()

    @unittest.skipIf(torch.cuda.device_count() < 2, "more than 1 GPU required")
    def test_multigpu(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.norm = InstanceNorm3dNVFuser(4)

            def forward(self, x):
                x = self.norm(x)
                x = torch.sum(x, dim=(1, 2, 3, 4))
                return x

        device = torch.device(f"cuda:1")
        model = Model().to(device)

        x = torch.randn(2, 4, 128, 128, 128, device=device, requires_grad=True)
        y = torch.randn(2, device=device)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y.float())
        loss.backward()


if __name__ == "__main__":
    unittest.main()
