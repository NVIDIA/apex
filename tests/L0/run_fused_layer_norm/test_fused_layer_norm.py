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
    def setUp(self):
        self.module_cpu_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 16], elementwise_affine=True).cpu()
        self.module_cuda_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 16], elementwise_affine=True).cuda()

