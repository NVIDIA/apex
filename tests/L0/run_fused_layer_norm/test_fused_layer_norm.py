import unittest
import os
import random

import torch
import apex
from torch.autograd import Variable


class TestFusedLayerNorm(unittest.TestCase):
    def setUp(self):
        # bias and weight are set to 0 and 1 respectively, so no need to copy parameters from cpu module to the gpu one
        self.module_cpu_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 16], elementwise_affine=False).cpu()
        self.module_cuda_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 16], elementwise_affine=False).cuda()

    def _test_same_output(self, batch_size):
        torch.cuda.manual_seed(42)
        self.input_ = torch.randn((batch_size, *self.module_cpu_.normalized_shape), device="cpu").requires_grad_(True)
        self.input_cuda_ = self.input_.cuda().detach().requires_grad_(True)
        out_cpu_ = self.module_cpu_(self.input_)
        gO = torch.rand_like(out_cpu_)
        out_cpu_.backward(gO)
        out_cuda_ = self.module_cuda_(self.input_cuda_)
        gO = gO.cuda()
        out_cuda_.backward(gO)
        assert out_cpu_.is_cuda == False
        assert out_cuda_.is_cuda == True
        torch.testing.assert_allclose(out_cpu_, out_cuda_.cpu())
        torch.testing.assert_allclose(self.input_.grad, self.input_cuda_.grad.cpu())

    def test_layer_norm(self):
        self._test_same_output(16)

    def test_large_batch(self):
        self._test_same_output(65536)


class TestFusedLayerNormElemWise(TestFusedLayerNorm):
    elementwise_affine = True


class TestFusedLayerNormElemWiseHalf(TestFusedLayerNormElemWise):
    dtype = torch.half

    def test_large_batch(self):
        self.skipTest("Skip to save time")


class TestFusedLayerNormElemWiseBFloat16(TestFusedLayerNormElemWise):
    dtype = torch.bfloat16
    # NOTE (mkozuki): [BFloat16 Layer Norm flakiness]
    # Use thresholds larger than those used in pytorch, see
    # https://github.com/pytorch/pytorch/blob/72274e2a2fd55019ec860e1743dbdc5b0c5a5624/torch/testing/_asserts.py#L26
    fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
    bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)

    def test_large_batch(self):
        self.skipTest("Skip to save time")


def _prep_layers(normalized_shape, elementwise_affine, dtype):
    native = torch.nn.LayerNorm(
        normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
    ).to(device="cuda", dtype=dtype)
    fused = apex.normalization.FusedLayerNorm(
        normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
    ).cuda()
    return native, fused


def _prep_inputs(batch_size, normalized_shape, dtype):
    shape = (batch_size, *normalized_shape)
    fused = torch.randn(shape).cuda().requires_grad_(True)
    with torch.no_grad():
        native = fused.clone().to(dtype).requires_grad_(True)
    return native, fused


autocast_dtypes = (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)


class TestAutocastFusedLayerNorm(unittest.TestCase):
    bf16_fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
    bf16_bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)

    def setUp(self):
        self.batch_size = 16
        self.normalized_shape = [32, 16]

    def _run_test(self, dtype, elementwise_affine):
        native, fused = _prep_layers(self.normalized_shape, elementwise_affine, dtype)
        native_x, fused_x = _prep_inputs(self.batch_size, self.normalized_shape, dtype)

        expected = native(native_x)
        with torch.cuda.amp.autocast(dtype=dtype):
            actual = fused(fused_x)
        tols = {'rtol': None, 'atol': None} if dtype == torch.half else TestAutocastFusedLayerNorm.bf16_fwd_thresholds
        torch.testing.assert_allclose(actual, expected, **tols)

        g_native = torch.rand_like(expected)
        with torch.no_grad():
            g_fused = g_native.clone()
        expected.backward(g_native)
        actual.backward(g_fused)


if __name__ == '__main__':
    unittest.main()
