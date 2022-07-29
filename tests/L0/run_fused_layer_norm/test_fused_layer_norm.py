import itertools
import unittest

import torch

import apex


class TestFusedLayerNorm(unittest.TestCase):
    dtype = torch.float
    elementwise_affine = False
    normalized_shape = [32, 16]
    rtol, atol = None, None
    fwd_thresholds = dict(rtol=None, atol=None)
    bwd_thresholds = dict(rtol=None, atol=None)
    mixed_fused = False

    def setUp(self):
        # bias and weight are set to 0 and 1 respectively, so no need to copy parameters from cpu module to the gpu one
        if not self.mixed_fused:
            self.module_cpu_ = apex.normalization.FusedLayerNorm(
                normalized_shape=self.normalized_shape, elementwise_affine=self.elementwise_affine).cpu()
            self.module_cuda_ = apex.normalization.FusedLayerNorm(
                normalized_shape=self.normalized_shape, elementwise_affine=self.elementwise_affine).to(device="cuda", dtype=self.dtype)
        else:
            assert self.elementwise_affine
            self.module_cpu_ = apex.normalization.MixedFusedLayerNorm(
                normalized_shape=self.normalized_shape).cpu()
            self.module_cuda_ = apex.normalization.MixedFusedLayerNorm(
                normalized_shape=self.normalized_shape).to(device="cuda", dtype=self.dtype)


    def _check_same_output(self, batch_size, contiguous):
        torch.cuda.manual_seed(42)
        if contiguous:
            input_shape = [batch_size] + self.normalized_shape
            input_ = torch.randn(input_shape, device="cpu").requires_grad_(True)
            input_cuda_ = input_.to(device="cuda", dtype=self.dtype).detach().requires_grad_(True)
            self.assertTrue(input_.is_contiguous())
            self.assertTrue(input_cuda_.is_contiguous())
        else:
            input_shape = [batch_size] + self.normalized_shape
            input_shape = [batch_size * 3] + [self.normalized_shape[0] * 5, self.normalized_shape[1] * 3]
            input_src_ = torch.randn(input_shape, device="cpu")
            input_ = input_src_[::3, ::5, ::3].detach().requires_grad_(True)
            input_cuda_ = input_src_.to(device="cuda", dtype=self.dtype)[::3, ::5, ::3].detach().requires_grad_(True)
            # make sure that tensors are NOT contiguous.
            self.assertFalse(input_.is_contiguous())
            self.assertFalse(input_cuda_.is_contiguous())
        out_cpu_ = self.module_cpu_(input_)
        gO = torch.rand_like(out_cpu_)
        out_cpu_.backward(gO)
        out_cuda_ = self.module_cuda_(input_cuda_)
        gO = gO.to(device="cuda", dtype=self.dtype)
        out_cuda_.backward(gO)
        self.assertFalse(out_cpu_.is_cuda)
        self.assertTrue(out_cuda_.is_cuda)
        # TODO (mkozuki): `torch.testing.assert_allclose` is deprecated.
        # Use `torch.testing.assert_close`.
        # See https://github.com/pytorch/pytorch/issues/61844
        torch.testing.assert_allclose(
            out_cpu_.to(device="cuda", dtype=self.dtype), out_cuda_, **self.fwd_thresholds)
        torch.testing.assert_allclose(
            input_.grad.to(device="cuda", dtype=self.dtype), input_cuda_.grad, **self.bwd_thresholds)

    def _test_same_output(self, batch_size):
        for contiguous in (True, False):
            with self.subTest(contiguous=contiguous):
                self._check_same_output(batch_size, contiguous)

    def test_layer_norm(self):
        self._test_same_output(16)

    def test_large_batch(self):
        self._test_same_output(65536)


class TestFusedRMSNorm(unittest.TestCase):
    dtype = torch.float
    elementwise_affine = False
    normalized_shape = [32, 16]
    rtol, atol = None, None
    fwd_thresholds = dict(rtol=None, atol=None)
    bwd_thresholds = dict(rtol=None, atol=None)
    mixed_fused = False

    def setUp(self):
        # bias and weight are set to 0 and 1 respectively, so no need to copy parameters from cpu module to the gpu one
        if not self.mixed_fused:
            self.module_cpu_ = apex.normalization.FusedRMSNorm(
                normalized_shape=self.normalized_shape, elementwise_affine=self.elementwise_affine).cpu()
            self.module_cuda_ = apex.normalization.FusedRMSNorm(
                normalized_shape=self.normalized_shape, elementwise_affine=self.elementwise_affine).to(device="cuda", dtype=self.dtype)
        else:
            assert self.elementwise_affine
            self.module_cpu_ = apex.normalization.MixedFusedRMSNorm(
                normalized_shape=self.normalized_shape).cpu()
            self.module_cuda_ = apex.normalization.MixedFusedRMSNorm(
                normalized_shape=self.normalized_shape).to(device="cuda", dtype=self.dtype)

    def _check_same_output(self, batch_size, contiguous):
        torch.cuda.manual_seed(42)
        if contiguous:
            input_shape = [batch_size] + self.normalized_shape
            input_ = torch.randn(input_shape, device="cpu").requires_grad_(True)
            input_cuda_ = input_.to(device="cuda", dtype=self.dtype).detach().requires_grad_(True)
            self.assertTrue(input_.is_contiguous())
            self.assertTrue(input_cuda_.is_contiguous())
        else:
            input_shape = [batch_size] + self.normalized_shape
            input_shape = [batch_size * 3] + [self.normalized_shape[0] * 5, self.normalized_shape[1] * 3]
            input_src_ = torch.randn(input_shape, device="cpu")
            input_ = input_src_[::3, ::5, ::3].detach().requires_grad_(True)
            input_cuda_ = input_src_.to(device="cuda", dtype=self.dtype)[::3, ::5, ::3].detach().requires_grad_(True)
            # make sure that tensors are NOT contiguous.
            self.assertFalse(input_.is_contiguous())
            self.assertFalse(input_cuda_.is_contiguous())
        out_cpu_ = self.module_cpu_(input_)
        gO = torch.rand_like(out_cpu_)
        out_cpu_.backward(gO)
        out_cuda_ = self.module_cuda_(input_cuda_)
        # TODO (mkozuki): `torch.testing.assert_allclose` is deprecated.
        # Use `torch.testing.assert_close`.
        # See https://github.com/pytorch/pytorch/issues/61844
        torch.testing.assert_allclose(
            out_cpu_.to(device="cuda", dtype=self.dtype), out_cuda_.clone().detach(), **self.fwd_thresholds)
        gO = gO.to(device="cuda", dtype=self.dtype)
        out_cuda_.backward(gO)
        self.assertFalse(out_cpu_.is_cuda)
        self.assertTrue(out_cuda_.is_cuda)
        torch.testing.assert_allclose(
            input_.grad.to(device="cuda", dtype=self.dtype), input_cuda_.grad, **self.bwd_thresholds)
        if self.elementwise_affine:
            torch.testing.assert_allclose(self.module_cpu_.weight.grad.to(device="cuda", dtype=self.dtype),
                                          self.module_cuda_.weight.grad, **self.bwd_thresholds)

    def _test_same_output(self, batch_size):
        for contiguous in (True, False):
            with self.subTest(contiguous=contiguous):
                self._check_same_output(batch_size, contiguous)

    def test_layer_norm(self):
        self._test_same_output(16)

    def test_large_batch(self):
        self._test_same_output(65536)


class TestFusedLayerNormElemWise(TestFusedLayerNorm):
    elementwise_affine = True

class TestMixedFusedLayerNormElemWise(TestFusedLayerNorm):
    elementwise_affine = True
    mixed_fused = True

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


class TestFusedRMSNormElemWise(TestFusedRMSNorm):
    bwd_thresholds = dict(rtol=2e-3, atol=2e-4)
    elementwise_affine = True

class TestMixedFusedRMSNormElemWise(TestFusedRMSNorm):
    bwd_thresholds = dict(rtol=2e-3, atol=2e-4)
    elementwise_affine = True
    mixed_fused = True

class TestFusedRMSNormElemWiseHalf(TestFusedRMSNormElemWise):
    dtype = torch.half
    bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)

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


def _prep_rms_layers(normalized_shape, elementwise_affine, dtype):
    native = apex.normalization.FusedRMSNorm(
        normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
    )
    fused = apex.normalization.FusedRMSNorm(
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

        tols = {'rtol': None, 'atol': None} if dtype == torch.half else TestAutocastFusedLayerNorm.bf16_bwd_thresholds
        torch.testing.assert_allclose(native_x.grad, fused_x.grad, **tols)

    def test_autocast(self):
        for (dtype, elementwise_affine) in itertools.product(autocast_dtypes, (True, False)):
            with self.subTest(f"{dtype}-{elementwise_affine}"):
                self._run_test(dtype, elementwise_affine)

class TestAutocastFusedRMSNorm(unittest.TestCase):
    bf16_fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
    bf16_bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)

    def setUp(self):
        self.batch_size = 16
        self.normalized_shape = [32, 16]

    def _run_test(self, dtype, elementwise_affine):
        native, fused = _prep_rms_layers(self.normalized_shape, elementwise_affine, dtype)
        native_x, fused_x = _prep_inputs(self.batch_size, self.normalized_shape, dtype)

        expected = native(native_x.cpu())
        with torch.cuda.amp.autocast(dtype=dtype):
            actual = fused(fused_x)
        tols = {'rtol': None, 'atol': None} if dtype == torch.half else TestAutocastFusedRMSNorm.bf16_fwd_thresholds
        torch.testing.assert_allclose(actual, expected.detach().clone().cuda(), **tols)

        g_native = torch.rand_like(expected)
        with torch.no_grad():
            g_fused = g_native.detach().clone().cuda()
        expected.backward(g_native)
        actual.backward(g_fused)

        tols = {'rtol': None, 'atol': None} if dtype == torch.half else TestAutocastFusedRMSNorm.bf16_bwd_thresholds
        torch.testing.assert_allclose(native_x.grad.cuda(), fused_x.grad, **tols)

    def test_autocast(self):
        for (dtype, elementwise_affine) in itertools.product(autocast_dtypes, (True, False)):
            with self.subTest(f"{dtype}-{elementwise_affine}"):
                self._run_test(dtype, elementwise_affine)
