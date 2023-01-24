import itertools
import unittest

from nvfuser import FusionCache

import torch
import torch.nn as nn

import apex
from apex.normalization import InstanceNorm3dNVFuser


def assert_close(a: torch.Tensor, b: torch.Tensor):
    """Given two Tensors, compare with a reasonable precision.

    If the dtypes mismatch, use a custom rule to cast one or the other
    """
    # increasing order of precision
    precedence = [torch.bfloat16, torch.float16, torch.float32, torch.float64]

    # demote inputs so we use the more permissive test
    if precedence.index(a.dtype) < precedence.index(b.dtype):
        b = b.type(a.dtype)
    else:
        a = a.type(b.dtype)

    if a.dtype in [torch.float16, torch.bfloat16]:
        # assert_close with high tolerance just checks device, dtype, layout, and stride
        torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
    else:  # use default tolerance
        torch.testing.assert_close(a, b)


class TestInstanceNormNVFuser(unittest.TestCase):
    dtype = torch.float
    track_running_stats = False
    channels_last = False
    affine = False
    batch_size = 5
    channel_size = 7
    spatial_size = 3

    def init_modules(self):
        # Uncomment below to verify that torch.nn.InstanceNorm3d passes our tests
        self.m = InstanceNorm3dNVFuser(
            # self.m = torch.nn.InstanceNorm3d(
            self.channel_size,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            device="cuda",
            dtype=self.dtype,
        )
        self.reference_m = torch.nn.InstanceNorm3d(
            self.channel_size,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            device="cuda",
            dtype=torch.float64,
        )

    def check_same_output(self, contiguous=True):
        torch.manual_seed(42)
        for i in range(2):  # exercise JIT + caching
            inp = torch.rand(
                (
                    self.batch_size,
                    self.channel_size,
                    self.spatial_size,
                    self.spatial_size,
                    self.spatial_size,
                ),
                device="cuda",
                requires_grad=True,
                dtype=self.dtype,
            )
            if self.channels_last:
                inp = inp.to(memory_format=torch.channels_last_3d)

            if not contiguous:
                inp = inp[..., ::2]

            inp = inp.detach()
            inp.requires_grad = True

            inp2 = inp.clone().type(torch.float64).detach()
            inp2.requires_grad = True

            assert (
                inp.is_contiguous(
                    memory_format=torch.channels_last_3d
                    if self.channels_last
                    else torch.contiguous_format
                )
                == contiguous
            )

            out = self.m(inp)
            out2 = self.reference_m(inp2)
            assert_close(out, out2)

            if self.m.running_mean is None:
                assert self.reference_m.running_mean is None
                assert self.m.running_var is None
                assert self.reference_m.running_var is None
            else:
                assert_close(self.m.running_mean, self.reference_m.running_mean)

            grad_out = torch.randn_like(inp)
            out.backward(grad_out)
            out2.backward(grad_out)
            assert_close(inp.grad, inp2.grad)

            # compare weight gradients
            if self.m.weight is not None:
                assert_close(self.m.weight.grad, self.reference_m.weight.grad)
            if self.m.bias is not None:
                assert_close(self.m.bias.grad, self.reference_m.bias.grad)

    def test_sweep(self):
        dtypes = [torch.float, torch.half, torch.double]
        if torch.cuda.get_device_capability() >= (8, 0):
            dtypes.append(torch.bfloat16)
        for dtype, track_running_stats, channels_last, affine in itertools.product(
            dtypes, (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                dtype=dtype,
                track_running_stats=track_running_stats,
                channels_last=channels_last,
                affine=affine,
            ):
                self.dtype = dtype
                self.track_running_stats = track_running_stats
                self.channels_last = channels_last
                self.affine = affine
                # observe the cache to see which params cause new Fusions
                fc = FusionCache.get()
                nf = fc.num_fusions()
                for b, c, s in [
                    (5, 7, 3),
                    (6, 7, 3),
                    (5, 8, 3),
                    # TODO: changing spatial size currently causes a new Fusion
                    # (5, 7, 4),
                    # (6, 7, 4),
                    # (5, 8, 4),
                ]:
                    for contig in [True, False]:
                        self.batch_size = b
                        self.channel_size = c
                        self.spatial_size = s

                        self.init_modules()
                        self.check_same_output(contig)
                    # Changing input sizes may cause a recompile, but it should
                    # not cause a new Fusion to be scheduled. However, changing
                    # contiguity should create a new Fusion. There is one
                    # Fusion for each of forward and backward, so 4 per subtest
                    assert fc.num_fusions() == nf + 4

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

        device = torch.device("cuda:1")
        model = Model().to(device)

        x = torch.randn(2, 4, 128, 128, 128, device=device, requires_grad=True)
        y = torch.randn(2, device=device)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y.float())
        loss.backward()


if __name__ == "__main__":
    unittest.main()
