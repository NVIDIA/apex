import unittest

import torch

from apex.normalization.fused_layer_norm import manual_rms_norm


class TestManualRMSNormPrecision(unittest.TestCase):
    def test_float64_values_and_gradients(self):
        for shape in ((4,), (2, 4)):
            for scale, eps in ((1.0, 1e-6), (1e20, 1e-6), (1e-25, 1e-55)):
                for affine in (False, True):
                    with self.subTest(shape=shape, scale=scale, affine=affine):
                        count = 4 if len(shape) == 1 else 8
                        values = torch.arange(1, 2 * count + 1, dtype=torch.float64)
                        values[::2] *= -1
                        x = (values.reshape(2, *shape) * scale).requires_grad_()
                        reference_x = x.detach().clone().requires_grad_()
                        weight = torch.linspace(0.5, 1.5, count, dtype=torch.float64).reshape(shape)
                        weight = weight.requires_grad_() if affine else None
                        reference_weight = (
                            weight.detach().clone().requires_grad_() if affine else None
                        )
                        actual = manual_rms_norm(x, shape, weight, eps)
                        expected = torch.nn.functional.rms_norm(
                            reference_x, shape, reference_weight, eps
                        )
                        cotangent = torch.linspace(
                            -0.7, 0.9, actual.numel(), dtype=torch.float64
                        ).reshape_as(actual)
                        actual.backward(cotangent)
                        expected.backward(cotangent)
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                        # Scaling avoids an absolute tolerance hiding tiny derivatives.
                        torch.testing.assert_close(
                            x.grad * scale, reference_x.grad * scale, rtol=1e-11, atol=1e-12
                        )
                        if affine:
                            torch.testing.assert_close(
                                weight.grad, reference_weight.grad, rtol=1e-12, atol=1e-12
                            )

    def test_low_precision_behavior_is_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for affine in (False, True):
                with self.subTest(dtype=dtype, affine=affine):
                    x = torch.linspace(-2, 3, 24, dtype=dtype).reshape(3, 2, 4).requires_grad_()
                    reference_x = x.detach().clone().requires_grad_()
                    weight = torch.linspace(0.5, 1.5, 8, dtype=dtype).reshape(2, 4)
                    weight = weight.requires_grad_() if affine else None
                    reference_weight = weight.detach().clone().requires_grad_() if affine else None
                    expected = reference_x * torch.rsqrt(
                        reference_x.float().square().mean((-1, -2), keepdim=True) + 1e-6
                    )
                    if reference_weight is not None:
                        if dtype in (torch.float16, torch.bfloat16):
                            expected = expected.to(dtype)
                        expected = expected * reference_weight
                    actual = manual_rms_norm(x, (2, 4), weight, 1e-6)
                    actual.sum().backward()
                    expected.sum().backward()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
                    if affine:
                        torch.testing.assert_close(
                            weight.grad, reference_weight.grad, rtol=0, atol=0
                        )

    def test_compiled_float64_matches_eager_reference(self):
        compiled = torch.compile(manual_rms_norm, backend="aot_eager", fullgraph=True)
        x = torch.tensor([[1e20, -2e20, 3e20, -4e20]], dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = compiled(x, (4,), None, 1e-6)
        expected = torch.nn.functional.rms_norm(reference_x, (4,), eps=1e-6)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(x.grad * 1e20, reference_x.grad * 1e20, rtol=1e-11, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
