import unittest

import functools as ft
import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_BFLOAT16, ALWAYS_FLOAT, MATCH_INPUT

from apex.testing.common_utils import skipIfRocm

def run_layer_test(test_case, fns, expected, input_shape, test_backward=True):
    for fn, typ in it.product(fns, expected.keys()):
        x = torch.randn(input_shape, dtype=typ).requires_grad_()
        y = fn(x)
        test_case.assertEqual(y.type(), expected[typ])
        if test_backward:
            y.float().sum().backward()
            test_case.assertEqual(x.grad.type(), MATCH_INPUT[typ])

class _TestBasicCasts(unittest.TestCase):
    def _test_linear(self, expected):
        m = nn.Linear(self.h, self.h)
        f = ft.partial(F.linear, weight=m.weight, bias=m.bias)
        run_layer_test(self, [m, f], expected, (self.b, self.h))

    def _test_conv2d(self, expected):
        m = nn.Conv2d(self.c, self.c, self.k)
        f = ft.partial(F.conv2d, weight=m.weight, bias=m.bias)
        run_layer_test(self, [m, f], expected, (self.b, self.c, self.h, self.h))

    def _test_softmax(self, expected):
        m = nn.Softmax(dim=1)
        f = ft.partial(F.softmax, dim=1)
        run_layer_test(self, [m, f], expected, (self.b, self.h))

    def _test_group_norm(self, expected):
        m = nn.GroupNorm(num_groups=4, num_channels=self.c)
        run_layer_test(self, [m], expected, (self.b, self.c, self.h, self.h))

    def _test_mse_loss(self, expected):
        shape = (self.b, self.h)
        target = torch.randn(shape)
        mod = nn.MSELoss()
        m = lambda x: mod(x, target)
        f = ft.partial(F.mse_loss, target=target)
        run_layer_test(self, [m], expected, shape)

    def _test_relu(self, expected):
        run_layer_test(self, [nn.ReLU(), F.relu], expected, (self.b, self.h))

    def _test_batch_norm(self, expected):
        m = nn.BatchNorm2d(num_features=self.c)
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=True)
        run_layer_test(self, [m], expected, (self.b, self.c, self.h, self.h))

        # Test forward-only for BN inference
        m.eval()
        f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m.running_var,
                       weight=m.weight, bias=m.bias, training=False)
        run_layer_test(self, [m, f], expected, (self.b, self.c, self.h, self.h),
                            test_backward=False)

class TestBasicCastsHalf(_TestBasicCasts):
    def setUp(self):
        self.handle = amp.init(enabled=True, patch_type=torch.half)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_linear_is_half(self):
        self._test_linear(ALWAYS_HALF)

    def test_conv2d_is_half(self):
        self._test_conv2d(ALWAYS_HALF)

    def test_softmax_is_float(self):
        self._test_softmax(ALWAYS_FLOAT)

    def test_group_norm_is_float(self):
        self._test_group_norm(ALWAYS_FLOAT)

    def test_mse_loss_is_float(self):
        self._test_mse_loss(ALWAYS_FLOAT)

    def test_relu_is_match(self):
        self._test_relu(MATCH_INPUT)

    def test_batch_norm_is_match(self):
        self._test_batch_norm(MATCH_INPUT)

class TestBasicCastsBFloat16(_TestBasicCasts):
    def setUp(self):
        self.handle = amp.init(enabled=True, patch_type=torch.bfloat16)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    @skipIfRocm
    def test_linear_is_bfloat16(self):
        self._test_linear(ALWAYS_BFLOAT16)

    @skipIfRocm
    def test_conv2d_is_bfloat16(self):
        self._test_conv2d(ALWAYS_BFLOAT16)

    def test_softmax_is_float(self):
        self._test_softmax(ALWAYS_FLOAT)

    def test_group_norm_is_float(self):
        self._test_group_norm(ALWAYS_FLOAT)

    def test_mse_loss_is_float(self):
        self._test_mse_loss(ALWAYS_FLOAT)

    def test_relu_is_match(self):
        self._test_relu(MATCH_INPUT)

    def test_batch_norm_is_match(self):
        self._test_batch_norm(MATCH_INPUT)

class TestBannedMethods(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True, patch_type=torch.half)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def bce_common(self, assertion, dtype=torch.half):
        shape = (self.b, self.h)
        target = torch.rand(shape)
        mod = nn.BCELoss()
        m = lambda x: mod(x, target)
        f = ft.partial(F.binary_cross_entropy, target=target)
        for fn in [m, f]:
            x = torch.rand(shape, dtype=dtype)
            assertion(fn, x)

    def test_bce_raises_by_default(self):
        assertion = lambda fn, x: self.assertRaises(NotImplementedError, fn, x)
        self.bce_common(assertion, dtype=torch.half)

        # handle with bfloat16 as patch_type
        self.handle._deactivate()
        self.handle = amp.init(enabled=True, patch_type=torch.bfloat16)
        self.bce_common(assertion, dtype=torch.bfloat16)

    def test_bce_is_float_with_allow_banned(self):
        self.handle._deactivate()
        self.handle = amp.init(enabled=True, allow_banned=True, patch_type=torch.half)
        assertion = lambda fn, x: self.assertEqual(fn(x).type(), FLOAT)
        self.bce_common(assertion, dtype=torch.half)

        # handle with bfloat16 as patch_type
        self.handle._deactivate()
        self.handle = amp.init(enabled=True, allow_banned=True, patch_type=torch.bfloat16)
        self.bce_common(assertion, dtype=torch.bfloat16)

class _TestTensorCasts(unittest.TestCase):
    def _test_matmul_method(self, expected):
        other = torch.randn(self.h, self.h)
        lhs = lambda x: x.matmul(other)
        rhs = lambda x: other.matmul(x)
        run_layer_test(self, [lhs, rhs], expected, (self.h, self.h))

    def _test_matmul_op(self, expected):
        other = torch.randn(self.h, self.h)
        lhs = lambda x: x @ other
        rhs = lambda x: other @ x
        run_layer_test(self, [lhs, rhs], expected, (self.h, self.h))

    def _test_pow_method(self, expected):
        fn = lambda x: x.pow(2.)
        run_layer_test(self, [fn], expected, (self.b, self.h))

    def _test_pow_op(self, expected):
        fn = lambda x: x ** 2.
        run_layer_test(self, [fn], expected, (self.b, self.h))

    def _test_cpu(self, expected):
        fn = lambda x: x.cpu()
        run_layer_test(self, [fn], expected, (self.b, self.h))

    def _test_sum(self, expected):
        fn = lambda x: x.sum()
        run_layer_test(self, [fn], expected, (self.b, self.h))

    # TODO: maybe more tests on disabled casting?

class TestTensorCastsHalf(_TestTensorCasts):
    def setUp(self):
        self.handle = amp.init(enabled=True, patch_type=torch.half)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def test_matmul_method_is_half(self):
        self._test_matmul_method(ALWAYS_HALF)

    def test_matmul_op_is_half(self):
        self._test_matmul_op(ALWAYS_HALF)

    def test_pow_method_is_float(self):
        self._test_pow_method(ALWAYS_FLOAT)

    def test_pow_op_is_float(self):
        self._test_pow_op(ALWAYS_FLOAT)

    def test_cpu_is_float(self):
        always_cpu_float = {torch.float: 'torch.FloatTensor',
                            torch.half: 'torch.FloatTensor'}
        self._test_cpu(always_cpu_float)

    def test_sum_is_float(self):
        self._test_sum(ALWAYS_FLOAT)

class TestTensorCastsBFloat16(_TestTensorCasts):
    def setUp(self):
        self.handle = amp.init(enabled=True, patch_type=torch.bfloat16)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    @skipIfRocm
    def test_matmul_method_is_bfloat16(self):
        self._test_matmul_method(ALWAYS_BFLOAT16)

    @skipIfRocm
    def test_matmul_op_is_bfloat16(self):
        self._test_matmul_op(ALWAYS_BFLOAT16)

    def test_pow_method_is_float(self):
        self._test_pow_method(ALWAYS_FLOAT)

    def test_pow_op_is_float(self):
        self._test_pow_op(ALWAYS_FLOAT)

    def test_cpu_is_float(self):
        always_cpu_float = {torch.float: 'torch.FloatTensor',
                            torch.bfloat16: 'torch.FloatTensor'}
        self._test_cpu(always_cpu_float)

    def test_sum_is_float(self):
        self._test_sum(ALWAYS_FLOAT)


if __name__ == '__main__':
    unittest.main()
