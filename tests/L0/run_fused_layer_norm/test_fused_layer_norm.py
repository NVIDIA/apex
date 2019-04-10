import unittest
import os
import random

import torch
import apex

        
class TestFusedLayerNorm(unittest.TestCase):
    def setUp(self):
        self.module = apex.normalization.FusedLayerNorm(normalized_shape=[32, 64], elementwise_affine=False)
        self.input_ = torch.randn(16, 32, 64)
        torch.cuda.manual_seed(42)
        
    def forward_cpu(self, input_):
        self.module.cpu()
        return self.module(input_.cpu())
    
    def forward_cuda(self, input_):
        self.module.cuda()
        return self.module(input_.cuda())
    
    def test_forward_cuda(self):
        out_ = self.forward_cuda(self.input_)
        assert out_.is_cuda == True
        
    def test_forward_cpu(self):
        out_ = self.forward_cpu(self.input_)
        assert out_.is_cuda == False
        
    def test_same_output(self):
        out_cpu = self.forward_cpu(self.input_)
        out_cuda = self.forward_cuda(self.input_)
        torch.testing.assert_allclose(out_cpu, out_cuda.cpu())
        
        
class TestFusedLayerNormElemWise(TestFusedLayerNorm):
    def setUp(self):
        self.module = apex.normalization.FusedLayerNorm(normalized_shape=[32, 64], elementwise_affine=True)
        self.input_ = torch.randn(16, 32, 64)
        torch.cuda.manual_seed(42)