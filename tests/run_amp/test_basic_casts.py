import unittest

import functools as ft
import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT

def run_layer_test(test_case, fns, expected, input_shape, test_backward=True):
    for fn, typ in it.product(fns, expected.keys()):
        x = torch.randn(input_shape, dtype=typ).requires_grad_()
        y = fn(x)
        test_case.assertEqual(y.type(), expected[typ])
        if test_backward:
            y.float().sum().backward()
            test_case.assertEqual(x.grad.type(), MATCH_INPUT[typ])

class TestBasicCasts(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_linear_is_half(self):
        m = nn.Linear(self.h, self.h)
        f = ft.partial(F.linear, weight=m.weight, bias=m.bias)
        run_layer_test(self, [m, f], ALWAYS_HALF, (self.b, self.h))

    def test_conv2d_is_half(self):
        m = nn.Conv2d(self.c, self.c, self.k)
        f = ft.partial(F.conv2d, weight=m.weight, bias=m.bias)
        run_layer_test(self, [m, f], ALWAYS_HALF, (self.b, self.c, self.h, self.h))

    def test_softmax_is_float(self):
        m = nn.Softmax(dim=1)
        f = ft.partial(F.softmax, dim=1)
        run_layer_test(self, [m, f], ALWAYS_FLOAT, (self.b, self.h))

    def test_group_norm_is_float(self):
        m = nn.GroupNorm(num_groups=4, num_channels=self.c)
        run_layer_test(self, [m], ALWAYS_FLOAT, (self.b, self.c, self.h, self.h))

    def test_mse_loss_is_float(self):
        shape = (self.b, self.h)
        target = torch.randn(shape)
        mod = nn.MSELoss()
        m = lambda x: mod(x, target)
        f = ft.partial(F.mse_loss, target=target)
        run_layer_test(self, [m], ALWAYS_FLOAT, shape)

    def test_relu_is_match(self):
        run_layer_test(self, [nn.ReLU(), F.relu], MATCH_INPUT, (self.b, self.h))

    def test_batch_norm_is_match(self):
        m = nn.BatchNorm2d(num_features=self.c)
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=True)
        run_layer_test(self, [m], MATCH_INPUT, (self.b, self.c, self.h, self.h))

        # Test forward-only for BN inference
        m.eval()
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=False)
        run_layer_test(self, [m, f], MATCH_INPUT, (self.b, self.c, self.h, self.h),
                            test_backward=False)

class TestBannedMethods(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def bce_common(self, assertion):
        shape = (self.b, self.h)
        target = torch.rand(shape)
        mod = nn.BCELoss()
        m = lambda x: mod(x, target)
        f = ft.partial(F.binary_cross_entropy, target=target)
        for fn in [m, f]:
            x = torch.rand(shape, dtype=torch.half)
            assertion(fn, x)

    def test_bce_raises_by_default(self):
        assertion = lambda fn, x: self.assertRaises(NotImplementedError, fn, x)
        self.bce_common(assertion)

    def test_bce_is_float_with_allow_banned(self):
        self.handle._deactivate()
        self.handle = amp.init(enabled=True, allow_banned=True)
        assertion = lambda fn, x: self.assertEqual(fn(x).type(), FLOAT)
        self.bce_common(assertion)

class TestTensorCasts(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_matmul_method_is_half(self):
        other = torch.randn(self.h, self.h)
        lhs = lambda x: x.matmul(other)
        rhs = lambda x: other.matmul(x)
        run_layer_test(self, [lhs, rhs], ALWAYS_HALF, (self.h, self.h))

    def test_matmul_op_is_half(self):
        other = torch.randn(self.h, self.h)
        lhs = lambda x: x @ other
        rhs = lambda x: other @ x
        run_layer_test(self, [lhs, rhs], ALWAYS_HALF, (self.h, self.h))

    def test_pow_method_is_float(self):
        fn = lambda x: x.pow(2.)
        run_layer_test(self, [fn], ALWAYS_FLOAT, (self.b, self.h))

    def test_pow_op_is_float(self):
        fn = lambda x: x ** 2.
        run_layer_test(self, [fn], ALWAYS_FLOAT, (self.b, self.h))

    def test_cpu_is_float(self):
        fn = lambda x: x.cpu()
        always_cpu_float = {torch.float: 'torch.FloatTensor',
                            torch.half: 'torch.FloatTensor'}
        run_layer_test(self, [fn], always_cpu_float, (self.b, self.h))

    def test_sum_is_float(self):
        fn = lambda x: x.sum()
        run_layer_test(self, [fn], ALWAYS_FLOAT, (self.b, self.h))

class TestDisabledCasts(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=False)
        common_init(self)

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
