import unittest

import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT, DTYPES

class TestPromotion(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def run_binary_promote_test(self, fns, input_shape, x_inplace=False):
        type_pairs = it.product(DTYPES, DTYPES)
        for fn, (xtype, ytype) in it.product(fns, type_pairs):
            x = torch.randn(input_shape, dtype=xtype).requires_grad_()
            x_leaf = x
            if x_inplace:
                # We need a non-leaf to call in place on
                x = x.clone()
            y = torch.randn(input_shape, dtype=ytype)
            out = fn(x, y)
            if x_inplace:
                # In place: always match xtype
                self.assertEqual(out.type(), x.type())
            else:
                # Out of place: match widest type
                if xtype == torch.float or ytype == torch.float:
                    self.assertEqual(out.type(), FLOAT)
                else:
                    self.assertEqual(out.type(), HALF)
            out.float().sum().backward()
            self.assertEqual(x_leaf.grad.dtype, xtype)

    def test_atan2_matches_widest(self):
        fns = [lambda x, y : torch.atan2(x, y),
               lambda x, y : x.atan2(y)]
        self.run_binary_promote_test(fns, (self.b,))

    def test_mul_matches_widest(self):
        fns = [lambda x, y : torch.mul(x, y),
               lambda x, y: x.mul(y)]
        self.run_binary_promote_test(fns, (self.b,))

    def test_cat_matches_widest(self):
        shape = self.b
        ys = [torch.randn(shape, dtype=torch.half) for _ in range(5)]
        x_float = torch.randn(shape)
        out = torch.cat(ys + [x_float])
        self.assertEqual(out.type(), FLOAT)
        x_half = torch.randn(shape, dtype=torch.half)
        out = torch.cat(ys + [x_half])
        self.assertEqual(out.type(), HALF)

    def test_inplace_exp_is_error_for_half(self):
        xs = torch.randn(self.b)
        xs.exp_()
        self.assertEqual(xs.type(), FLOAT)
        xs = torch.randn(self.b, dtype=torch.half)
        with self.assertRaises(NotImplementedError):
            xs.exp_()

    def test_inplace_add_matches_self(self):
        fn = lambda x, y: x.add_(y)
        self.run_binary_promote_test([fn], (self.b,), x_inplace=True)

if __name__ == '__main__':
    unittest.main()
