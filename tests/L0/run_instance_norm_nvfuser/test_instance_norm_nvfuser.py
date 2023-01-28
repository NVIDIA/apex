import itertools
import unittest

import pytest

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
        # torch.nn.InstanceNorm3d fails rtol=6, atols=4e-2 for half precision
        torch.testing.assert_close(a, b, rtol=10, atol=5e-2)
    else:  # use default tolerance
        torch.testing.assert_close(a, b)


dtypes = {
    "float32": torch.float,
    "float64": torch.double,
    "float16": torch.half,
}
if torch.cuda.get_device_capability() >= (8, 0):
    dtypes["bfloat16"] = torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,channel_size,spatial_size,compare",
    [
        (5, 7, 3, True),
        # check size=1 dimensions
        (1, 7, 3, True),  # NOTE: FAILS!
        (5, 1, 3, True),
        # (5, 7, 1, True), # eager instance norm needs more than one spatial element
        (1, 1, 3, True),
        # Don't check output for larger inputs, but check that they run
        # (16, 1, 64, False),
        # (16, 2, 64, False),
        # (1, 16, 64, False),
        # (2, 16, 64, False),
        # (16, 16, 64, False),
    ],
)
@pytest.mark.parametrize("memory_format", ["contiguous", "channels_last", "strided"])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("track_running_stats", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", dtypes.keys())
def test_match(
    dtype,
    training,
    track_running_stats,
    memory_format,
    affine,
    batch_size,
    channel_size,
    spatial_size,
    compare,
):
    dtype = dtypes[dtype]
    m = InstanceNorm3dNVFuser(
        channel_size,
        affine=affine,
        track_running_stats=track_running_stats,
        device="cuda",
        dtype=dtype,
    )
    reference_m = torch.nn.InstanceNorm3d(
        channel_size,
        affine=affine,
        track_running_stats=track_running_stats,
        device="cuda",
        dtype=torch.float64,
    )
    torch.manual_seed(42)
    for i in range(2):  # exercise JIT + caching
        inp = torch.rand(
            (
                batch_size,
                channel_size,
                spatial_size,
                spatial_size,
                2 * spatial_size if memory_format == "strided" else spatial_size,
            ),
            device="cuda",
            requires_grad=True,
            dtype=dtype,
        )
        if memory_format == "channels_last":
            inp = inp.to(memory_format=torch.channels_last_3d)
        elif memory_format == "strided":
            inp = inp[..., ::2]

        inp = inp.detach()
        inp.requires_grad = True

        inp2 = inp.clone().type(torch.float64).detach()
        inp2.requires_grad = True

        if training:
            m.train()
            reference_m.train()
        else:
            m.eval()
            reference_m.eval()

        out = m(inp)
        out2 = reference_m(inp2)
        if compare:
            assert_close(out, out2)

        if m.running_mean is None:
            assert reference_m.running_mean is None
            assert m.running_var is None
            assert reference_m.running_var is None
        else:
            if compare:
                assert_close(m.running_mean, reference_m.running_mean)

        if not training:
            return

        grad_out = torch.randn_like(inp)
        out.backward(grad_out)
        out2.backward(grad_out)
        if compare:
            assert_close(inp.grad, inp2.grad)

            # compare weight gradients
            if m.weight is not None:
                assert_close(m.weight.grad, reference_m.weight.grad)
            if m.bias is not None:
                assert_close(m.bias.grad, reference_m.bias.grad)


@unittest.skipIf(torch.cuda.device_count() < 2, "more than 1 GPU required")
def test_multigpu():
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
    pytest.main(["-v", __file__])
