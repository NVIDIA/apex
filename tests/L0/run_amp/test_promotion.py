import unittest

import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT, DTYPES, DTYPES2, MATCH_INPUT

class _TestPromotion(unittest.TestCase):
    def run_binary_promote_test(self, fns, input_shape, lp_type, x_inplace=False):
        if lp_type == torch.half:
            dtypes = DTYPES
        elif lp_type == torch.bfloat16:
            dtypes = DTYPES2
        else:
            raise RuntimeError("Creating test class with invalid low_precision type. \
                                Supported types are torch.half and torch.bfloat16")
        type_pairs = it.product(dtypes, dtypes)
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
                    self.assertEqual(out.type(), MATCH_INPUT[lp_type])
            out.float().sum().backward()
            self.assertEqual(x_leaf.grad.dtype, xtype)

    def _test_cat_matches_widest(self, lp_type):
        shape = self.b
        ys = [torch.randn(shape, dtype=lp_type) for _ in range(5)]
        x_float = torch.randn(shape)
        out = torch.cat(ys + [x_float])
        self.assertEqual(out.type(), FLOAT)
        x_lp = torch.randn(shape, dtype=lp_type)
        out = torch.cat(ys + [x_lp])
        self.assertEqual(out.type(), MATCH_INPUT[lp_type])

    def _test_inplace_exp_is_error_for_lp(self, lp_type):
        xs = torch.randn(self.b)
        xs.exp_()
        self.assertEqual(xs.type(), FLOAT)
        xs = torch.randn(self.b, dtype=lp_type)
        with self.assertRaises(NotImplementedError):
            xs.exp_()

class TestPromotionHalf(_TestPromotion):
    def setUp(self):
        self.handle = amp.init(enabled=True, patch_type=torch.half)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_atan2_matches_widest(self):
        fns = [lambda x, y : torch.atan2(x, y),
               lambda x, y : x.atan2(y)]
        self.run_binary_promote_test(fns, (self.b,), torch.half)

    def test_mul_matches_widest(self):
        fns = [lambda x, y : torch.mul(x, y),
               lambda x, y: x.mul(y)]
        self.run_binary_promote_test(fns, (self.b,), torch.half)

    def test_cat_matches_widest(self):
        self._test_cat_matches_widest(torch.half)

    def test_inplace_exp_is_error_for_half(self):
        self._test_inplace_exp_is_error_for_lp(torch.half)

    def test_inplace_add_matches_self(self):
        fn = lambda x, y: x.add_(y)
        self.run_binary_promote_test([fn], (self.b,), torch.half, x_inplace=True)

class TestPromotionBFloat16(_TestPromotion):
    def setUp(self):
        self.handle = amp.init(enabled=True, patch_type=torch.bfloat16)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_mul_matches_widest(self):
        fns = [lambda x, y : torch.mul(x, y),
               lambda x, y: x.mul(y)]
        self.run_binary_promote_test(fns, (self.b,), torch.bfloat16)

    def test_cat_matches_widest(self):
        self._test_cat_matches_widest(torch.bfloat16)

    def test_inplace_exp_is_error_for_bfloat16(self):
        self._test_inplace_exp_is_error_for_lp(torch.bfloat16)

    def test_inplace_add_matches_self(self):
        fn = lambda x, y: x.add_(y)
        self.run_binary_promote_test([fn], (self.b,), torch.bfloat16, x_inplace=True)

if __name__ == '__main__':
    unittest.main()
