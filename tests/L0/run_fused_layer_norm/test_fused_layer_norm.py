import unittest
import os
import random

import torch
import apex
from torch.autograd import Variable

        
class TestFusedLayerNorm(unittest.TestCase):
    def setUp(self):
        self.module_cpu_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 64], elementwise_affine=False).cpu()
        self.module_cuda_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 64], elementwise_affine=False).cuda()
        torch.cuda.manual_seed(42)
        self.input_ = torch.randn(16, 32, 64)
        self.loss_ = torch.randn(16, 32, 64)
        self.input_cpu_ = Variable(self.input_.detach().clone(), requires_grad=True)
        self.loss_cpu_ = self.loss_.detach().clone()
        self.input_cuda_ = Variable(self.input_.detach().clone().cuda(), requires_grad=True)
        self.loss_cuda_ = self.loss_.detach().clone().cuda()

    def test_same_output(self):
        out_cpu_ = self.module_cpu_(self.input_cpu_)
        out_cpu_.backward(self.loss_cpu_)
        out_cuda_ = self.module_cuda_(self.input_cuda_)
        out_cuda_.backward(self.loss_cuda_)
        assert out_cpu_.is_cuda == False
        assert out_cuda_.is_cuda == True
        torch.testing.assert_allclose(out_cpu_, out_cuda_.cpu())
        torch.testing.assert_allclose(self.input_cpu_.grad, self.input_cuda_.grad.cpu())
        
        
class TestFusedLayerNormElemWise(TestFusedLayerNorm):
    def setUp(self):
        self.module_cpu_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 64], elementwise_affine=True).cpu()
        self.module_cuda_ = apex.normalization.FusedLayerNorm(normalized_shape=[32, 64], elementwise_affine=True).cuda()
        torch.cuda.manual_seed(42)
        self.input_ = torch.randn(16, 32, 64)
        self.loss_ = torch.randn(16, 32, 64)
        self.input_cpu_ = Variable(self.input_.detach().clone(), requires_grad=True)
        self.loss_cpu_ = self.loss_.detach().clone()
        self.input_cuda_ = Variable(self.input_.detach().clone().cuda(), requires_grad=True)
        self.loss_cuda_ = self.loss_.detach().clone().cuda()
