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

    def setUp(self):
        # bias and weight are set to 0 and 1 respectively, so no need to copy parameters from cpu module to the gpu one
        self.module_cpu_ = apex.normalization.FusedLayerNorm(
            normalized_shape=self.normalized_shape, elementwise_affine=self.elementwise_affine).cpu()
        self.module_cuda_ = apex.normalization.FusedLayerNorm(
            normalized_shape=self.normalized_shape, elementwise_affine=self.elementwise_affine).to(device="cuda", dtype=self.dtype)

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


class TestFusedLayerNormElemWise(TestFusedLayerNorm):
    elementwise_affine = True


class TestFusedLayerNormElemWiseHalf(TestFusedLayerNormElemWise):
    dtype = torch.half

    def test_large_batch(self):
        self.skipTest("Skip to save time")


# Megatron style Layer Norm
class TestFusedLayerNormElemWiseMixedDtypes(TestFusedLayerNorm):
    def setUp(self):
        self.module_cpu_ = apex.normalization.MixedFusedLayerNorm(
            normalized_shape=self.normalized_shape, elementwise_affine=True).cpu()
        self.module_cuda_ = apex.normalization.MixedFusedLayerNorm(
            normalized_shape=self.normalized_shape, elementwise_affine=True).to(device="cuda", dtype=self.dtype)

    def test_init_exception(self):
        with self.assertRaisesRegex(RuntimeError, "MixedFusedLayerNorm does not support `elementwise_affine = False`"):
            apex.normalization.MixedFusedLayerNorm(normalized_shape=[32, 16], elementwise_affine=False).cuda()


class TestFusedLayerNormElemWiseMixedDtypesHalf(TestFusedLayerNormElemWiseMixedDtypes):
    dtype = torch.half

    def test_large_batch(self):
        self.skipTest("Skip to save time")


# NOTE (mkozuki): With the larger threshold values, still flaky.
class TestFusedLayerNormElemWiseMixedDtypesBFloat16(TestFusedLayerNormElemWiseMixedDtypesHalf):
    dtype = torch.bfloat16
    # NOTE (mkozuki): [BFloat16 Layer Norm flakiness]
    # Use thresholds larger than those used in pytorch, see
    # https://github.com/pytorch/pytorch/blob/72274e2a2fd55019ec860e1743dbdc5b0c5a5624/torch/testing/_asserts.py#L26
    fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
    bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)


class TestFusedLayerNormElemWiseBFloat16(TestFusedLayerNormElemWise):
    dtype = torch.bfloat16
    # See [BFloat16 Layer Norm flakiness]
    fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
    bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)

    def test_large_batch(self):
        self.skipTest("Skip to save time")
