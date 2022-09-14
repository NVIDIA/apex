import random
import unittest

import torch

HAS_INDEX_MUL_2D_RELU = None
try:
    from apex.contrib.index_mul_2d import index_mul_2d
except ImportError as e:
    HAS_INDEX_MUL_2D_RELU = False
else:
    HAS_INDEX_MUL_2D_RELU = True


@unittest.skipIf(not HAS_INDEX_MUL_2D_RELU, "`apex.contrib.index_mul_2d` is not found.")
class IndexMul2dTest(unittest.TestCase):
    def setUp(self, seed=0):
        torch.manual_seed(seed)

        self.input1_size = random.randint(1, 1000)
        self.input2_size = random.randint(1, 100000)
        self.feature_size = random.randint(1, 256)

        self.input1_float = torch.randn(size=(self.input1_size, self.feature_size),).cuda()
        self.input2_float = torch.randn(size=(self.input2_size, self.feature_size),).cuda()
        self.index1 = torch.randint(low=0, high=self.input1_size, size=(self.input2_size,)).cuda()

        self.input1_float_ = self.input1_float.clone()
        self.input2_float_ = self.input2_float.clone()

        self.input1_float.requires_grad_()
        self.input1_float_.requires_grad_()
        self.input2_float.requires_grad_()
        self.input2_float_.requires_grad_()

        self.input1_half = torch.randn(size=(self.input1_size, self.feature_size),).cuda().half()
        self.input2_half = torch.randn(size=(self.input2_size, self.feature_size),).cuda().half()

        self.input1_half_ = self.input1_half.clone()
        self.input2_half_ = self.input2_half.clone()

        self.input1_half.requires_grad_()
        self.input2_half.requires_grad_()
        self.input1_half_.requires_grad_()
        self.input2_half_.requires_grad_()

    def test_index_mul_float(self):
        out = index_mul_2d(self.input1_float, self.input2_float, self.index1)
        energy = (out.float()**2).sum() / out.numel()
        force = torch.autograd.grad(
            energy,
            self.input1_float,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
        )[0]
        loss = (out.float()**2).sum() / out.numel() + (force.float()**2).sum()
        loss.backward()

        out_ = self.input1_float_[self.index1] * self.input2_float_
        energy_ = (out_.float()**2).sum() / out.numel()
        force_ = torch.autograd.grad(
            energy_,
            self.input1_float_,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
        )[0]
        loss = (out_.float()**2).sum() / out_.numel() + (force_.float()**2).sum()
        loss.backward()

        self.assertTrue(torch.allclose(self.input1_float, self.input1_float_, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.input2_float, self.input2_float_, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.input1_float.grad, self.input1_float_.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.input2_float.grad, self.input2_float_.grad, atol=1e-3, rtol=1e-3, equal_nan=True))

    def test_index_mul_half(self):
        out = index_mul_2d(self.input1_half, self.input2_half, self.index1)
        energy = (out.float()**2).sum() / out.numel()
        force = torch.autograd.grad(
            energy,
            self.input1_half,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
        )[0]
        loss = (out.float()**2).sum() / out.numel() + (force.float()**2).sum()
        loss.backward()

        out_ = self.input1_half_[self.index1] * self.input2_half_
        energy_ = (out_.float()**2).sum() / out.numel()
        force_ = torch.autograd.grad(
            energy_,
            self.input1_half_,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
        )[0]
        loss = (out_.float()**2).sum() / out_.numel() + (force_.float()**2).sum()
        loss.backward()

        self.assertTrue(torch.allclose(self.input1_half, self.input1_half_, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.input2_half, self.input2_half_, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.input1_half.grad, self.input1_half_.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(self.input2_half.grad, self.input2_half_.grad, atol=1e-3, rtol=1e-3, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
