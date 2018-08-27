import unittest

import functools as ft
import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

HALF = 'torch.cuda.HalfTensor'
FLOAT = 'torch.cuda.FloatTensor'

ALWAYS_HALF = {torch.float: HALF,
               torch.half: HALF}
ALWAYS_FLOAT = {torch.float: FLOAT,
                torch.half: FLOAT}
MATCH_INPUT = {torch.float: FLOAT,
               torch.half: HALF}

def _common_init(test_case):
    test_case.h = 64
    test_case.b = 16
    test_case.c = 16
    test_case.k = 3
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class TestBasicCasts(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        _common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def run_layer_test(self, fns, expected, input_shape, test_backward=True):
        for fn, typ in it.product(fns, expected.keys()):
            x = torch.randn(input_shape, dtype=typ).requires_grad_()
            y = fn(x)
            self.assertEqual(y.type(), expected[typ])
            if test_backward:
                y.float().sum().backward()
                self.assertEqual(x.grad.type(), MATCH_INPUT[typ])

    def test_linear_is_half(self):
        m = nn.Linear(self.h, self.h)
        f = ft.partial(F.linear, weight=m.weight, bias=m.bias)
        self.run_layer_test([m, f], ALWAYS_HALF, (self.b, self.h))

    def test_conv2d_is_half(self):
        m = nn.Conv2d(self.c, self.c, self.k)
        f = ft.partial(F.conv2d, weight=m.weight, bias=m.bias)
        self.run_layer_test([m, f], ALWAYS_HALF, (self.b, self.c, self.h, self.h))

    def test_softmax_is_float(self):
        m = nn.Softmax(dim=1)
        f = ft.partial(F.softmax, dim=1)
        self.run_layer_test([m, f], ALWAYS_FLOAT, (self.b, self.h))

    def test_group_norm_is_float(self):
        m = nn.GroupNorm(num_groups=4, num_channels=self.c)
        self.run_layer_test([m], ALWAYS_FLOAT, (self.b, self.c, self.h, self.h))

    def test_mse_loss_is_float(self):
        shape = (self.b, self.h)
        target = torch.randn(shape)
        mod = nn.MSELoss()
        m = lambda x: mod(x, target)
        f = ft.partial(F.mse_loss, target=target)
        self.run_layer_test([m], ALWAYS_FLOAT, shape)

    def test_relu_is_match(self):
        self.run_layer_test([nn.ReLU(), F.relu], MATCH_INPUT, (self.b, self.h))

    def test_batch_norm_is_match(self):
        m = nn.BatchNorm2d(num_features=self.c)
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=True)
        self.run_layer_test([m], MATCH_INPUT, (self.b, self.c, self.h, self.h))

        # Test forward-only for BN inference
        m.eval()
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=False)
        self.run_layer_test([m, f], MATCH_INPUT, (self.b, self.c, self.h, self.h), test_backward=False)

class TestDisabledCasts(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=False)
        _common_init(self)

    def test_disabled_linear(self):
        m = nn.Linear(self.h, self.h)
        f = ft.partial(F.linear, weight=m.weight, bias=m.bias)
        input_shape = (self.b, self.h)

        for fn in [m, f]:
            x = torch.randn(input_shape, dtype=torch.float).requires_grad_()
            y = fn(x)
            self.assertEqual(y.type(), FLOAT)
            y.sum().backward()
            self.assertEqual(x.grad.type(), FLOAT)

            x = torch.randn(input_shape, dtype=torch.half).requires_grad_()
            self.assertRaises(RuntimeError, fn, x)

    # TODO: maybe more tests on disabled casting?

if __name__ == '__main__':
    unittest.main()
