import unittest

import torch
from torch.testing._internal.common_device_type import largeTensorTest

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    HAS_APEX = True
except ImportError:
    HAS_APEX = False


@unittest.skipIf(not HAS_APEX, "`amp_C` is not found.")
class MultiTensorScaleTest(unittest.TestCase):
    def testFP32(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        src = torch.full((1024,), 2.0, dtype=torch.float, device="cuda")
        dst = torch.zeros_like(src)
        multi_tensor_applier(amp_C.multi_tensor_scale, noop_flag, [[src], [dst]], 0.5)
        torch.testing.assert_close(dst, torch.ones(1024, device="cuda"))

    def testFP16toFP32(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        src = torch.full((1024,), 4.0, dtype=torch.half, device="cuda")
        dst = torch.zeros(1024, dtype=torch.float, device="cuda")
        multi_tensor_applier(amp_C.multi_tensor_scale, noop_flag, [[src], [dst]], 0.25)
        torch.testing.assert_close(dst, torch.ones(1024, device="cuda"))

    def testMultiTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        srcs = [torch.full((s,), 3.0, dtype=torch.float, device="cuda") for s in [1024, 2048, 512]]
        dsts = [torch.zeros_like(s) for s in srcs]
        multi_tensor_applier(amp_C.multi_tensor_scale, noop_flag, [srcs, dsts], 2.0)
        for dst in dsts:
            torch.testing.assert_close(dst, torch.full_like(dst, 6.0))

    @largeTensorTest("60GB", "cuda")
    def testLargeTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        src = torch.full((2359332864,), 2.0, dtype=torch.float, device="cuda")
        dst = torch.zeros_like(src)
        multi_tensor_applier(amp_C.multi_tensor_scale, noop_flag, [[src], [dst]], 0.5)
        torch.testing.assert_close(dst[-1:], torch.tensor([1.0], device="cuda"))
        torch.testing.assert_close(dst[:1], torch.tensor([1.0], device="cuda"))

    @largeTensorTest("60GB", "cuda")
    def testLargeTensorHalf(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        src = torch.full((2359332864,), 2.0, dtype=torch.half, device="cuda")
        dst = torch.zeros(2359332864, dtype=torch.float, device="cuda")
        multi_tensor_applier(amp_C.multi_tensor_scale, noop_flag, [[src], [dst]], 0.5)
        torch.testing.assert_close(dst[-1:], torch.tensor([1.0], device="cuda"))


@unittest.skipIf(not HAS_APEX, "`amp_C` is not found.")
class MultiTensorL2NormTest(unittest.TestCase):
    def testFP32(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(1024, dtype=torch.float, device="cuda")
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, noop_flag, [[t]], False)
        expected = torch.tensor([1024.0**0.5], device="cuda")
        torch.testing.assert_close(norm, expected)

    def testFP16(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(1024, dtype=torch.half, device="cuda")
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, noop_flag, [[t]], False)
        expected = torch.tensor([1024.0**0.5], device="cuda")
        torch.testing.assert_close(norm, expected)

    def testMultiTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        tensors = [torch.ones(s, dtype=torch.float, device="cuda") for s in [1024, 2048, 512]]
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, noop_flag, [tensors], False)
        expected = torch.tensor([(1024 + 2048 + 512) ** 0.5], device="cuda")
        torch.testing.assert_close(norm, expected)

    def testUnscale(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(1024, dtype=torch.float, device="cuda")
        inv_scale = torch.tensor([0.5], dtype=torch.float, device="cuda")
        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_unscale_l2norm, noop_flag, [[t]], inv_scale, False
        )
        expected = torch.tensor([1024.0**0.5 * 0.5], device="cuda")
        torch.testing.assert_close(norm, expected)

    @largeTensorTest("60GB", "cuda")
    def testLargeTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(2359332864, dtype=torch.float, device="cuda")
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, noop_flag, [[t]], False)
        expected = torch.tensor([2359332864.0**0.5], device="cuda")
        # Hierarchical float32 reduction over 2.3B elements introduces accumulation error
        torch.testing.assert_close(norm, expected, atol=1.0, rtol=1e-5)

    @largeTensorTest("60GB", "cuda")
    def testLargeTensorHalf(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(2359332864, dtype=torch.half, device="cuda")
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm, noop_flag, [[t]], False)
        expected = torch.tensor([2359332864.0**0.5], device="cuda")
        torch.testing.assert_close(norm, expected, atol=256.0, rtol=1e-2)

    @largeTensorTest("60GB", "cuda")
    def testUnscaleLargeTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(2359332864, dtype=torch.float, device="cuda")
        inv_scale = torch.tensor([0.5], dtype=torch.float, device="cuda")
        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_unscale_l2norm, noop_flag, [[t]], inv_scale, False
        )
        expected = torch.tensor([2359332864.0**0.5 * 0.5], device="cuda")
        torch.testing.assert_close(norm, expected, atol=1.0, rtol=1e-5)


@unittest.skipIf(not HAS_APEX, "`amp_C` is not found.")
class MultiTensorSGDTest(unittest.TestCase):
    def testBasic(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        lr = 0.01

        grad = torch.ones(1024, dtype=torch.float, device="cuda")
        weight = torch.ones(1024, dtype=torch.float, device="cuda")
        mom = torch.zeros(1024, dtype=torch.float, device="cuda")

        multi_tensor_applier(
            amp_C.multi_tensor_sgd,
            noop_flag,
            [[grad], [weight], [mom]],
            0.0,
            0.9,
            0.0,
            lr,
            False,
            True,
            False,
            1.0,
        )
        expected = torch.full((1024,), 1.0 - lr, device="cuda")
        torch.testing.assert_close(weight, expected)

    def testMultiTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        lr = 0.1

        grads = [torch.ones(s, dtype=torch.float, device="cuda") for s in [1024, 2048]]
        weights = [torch.ones(s, dtype=torch.float, device="cuda") for s in [1024, 2048]]
        moms = [torch.zeros(s, dtype=torch.float, device="cuda") for s in [1024, 2048]]

        multi_tensor_applier(
            amp_C.multi_tensor_sgd,
            noop_flag,
            [grads, weights, moms],
            0.0,
            0.9,
            0.0,
            lr,
            False,
            True,
            False,
            1.0,
        )
        for w in weights:
            torch.testing.assert_close(w, torch.full_like(w, 1.0 - lr))

    @largeTensorTest("60GB", "cuda")
    def testLargeTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        lr = 0.01

        grad = torch.ones(2359332864, dtype=torch.float, device="cuda")
        weight = torch.ones(2359332864, dtype=torch.float, device="cuda")
        mom = torch.zeros(2359332864, dtype=torch.float, device="cuda")

        multi_tensor_applier(
            amp_C.multi_tensor_sgd,
            noop_flag,
            [[grad], [weight], [mom]],
            0.0,
            0.9,
            0.0,
            lr,
            False,
            True,
            False,
            1.0,
        )
        expected = torch.tensor([1.0 - lr], device="cuda")
        torch.testing.assert_close(weight[-1:], expected)
        torch.testing.assert_close(weight[:1], expected)


@unittest.skipIf(not HAS_APEX, "`amp_C` is not found.")
class MultiTensorAxpbyTest(unittest.TestCase):
    def testBasic(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        x = torch.full((1024,), 2.0, dtype=torch.float, device="cuda")
        y = torch.full((1024,), 3.0, dtype=torch.float, device="cuda")
        out = torch.zeros(1024, dtype=torch.float, device="cuda")
        # out = a*x + b*y = 0.5*2 + 0.25*3 = 1.75
        multi_tensor_applier(amp_C.multi_tensor_axpby, noop_flag, [[x], [y], [out]], 0.5, 0.25, -1)
        torch.testing.assert_close(out, torch.full((1024,), 1.75, device="cuda"))

    def testMultiTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        xs = [torch.full((s,), 1.0, dtype=torch.float, device="cuda") for s in [1024, 2048]]
        ys = [torch.full((s,), 2.0, dtype=torch.float, device="cuda") for s in [1024, 2048]]
        outs = [torch.zeros(s, dtype=torch.float, device="cuda") for s in [1024, 2048]]
        # out = 2*x + 3*y = 2*1 + 3*2 = 8
        multi_tensor_applier(amp_C.multi_tensor_axpby, noop_flag, [xs, ys, outs], 2.0, 3.0, -1)
        for out in outs:
            torch.testing.assert_close(out, torch.full_like(out, 8.0))

    @largeTensorTest("60GB", "cuda")
    def testLargeTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        numel = 2359332864
        x = torch.full((numel,), 2.0, dtype=torch.float, device="cuda")
        y = torch.full((numel,), 3.0, dtype=torch.float, device="cuda")
        out = torch.zeros(numel, dtype=torch.float, device="cuda")
        multi_tensor_applier(amp_C.multi_tensor_axpby, noop_flag, [[x], [y], [out]], 0.5, 0.25, -1)
        # out = 0.5*2 + 0.25*3 = 1.75 (exact in float32)
        expected = torch.full((numel,), 1.75, device="cuda")
        torch.testing.assert_close(out, expected)


@unittest.skipIf(not HAS_APEX, "`amp_C` is not found.")
class MultiTensorL2NormMPTest(unittest.TestCase):
    def testFP32(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(1024, dtype=torch.float, device="cuda")
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm_mp, noop_flag, [[t]], False)
        expected = torch.tensor([1024.0**0.5], device="cuda")
        torch.testing.assert_close(norm, expected)

    def testBF16(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm_mp, noop_flag, [[t]], False)
        expected = torch.tensor([1024.0**0.5], device="cuda")
        torch.testing.assert_close(norm, expected)

    @largeTensorTest("60GB", "cuda")
    def testLargeTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        t = torch.ones(2359332864, dtype=torch.float, device="cuda")
        norm, _ = multi_tensor_applier(amp_C.multi_tensor_l2norm_mp, noop_flag, [[t]], False)
        expected = torch.tensor([2359332864.0**0.5], device="cuda")
        torch.testing.assert_close(norm, expected, atol=1.0, rtol=1e-5)


@unittest.skipIf(not HAS_APEX, "`amp_C` is not found.")
class MultiTensorL2NormScaleTest(unittest.TestCase):
    def testBasic(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        src = torch.ones(1024, dtype=torch.float, device="cuda")
        dst = torch.zeros(1024, dtype=torch.float, device="cuda")
        scale = 2.0
        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm_scale, noop_flag, [[src], [dst]], scale, False
        )
        # norm = sqrt(sum(1^2)) = sqrt(1024)
        expected_norm = torch.tensor([1024.0**0.5], device="cuda")
        torch.testing.assert_close(norm, expected_norm)
        # dst = src * scale = 2.0
        torch.testing.assert_close(dst, torch.full((1024,), 2.0, device="cuda"))

    @largeTensorTest("60GB", "cuda")
    def testLargeTensor(self):
        noop_flag = torch.tensor([0], dtype=torch.int, device="cuda")
        numel = 2359332864
        src = torch.ones(numel, dtype=torch.float, device="cuda")
        dst = torch.zeros(numel, dtype=torch.float, device="cuda")
        scale = 0.5
        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm_scale, noop_flag, [[src], [dst]], scale, False
        )
        expected_norm = torch.tensor([numel**0.5], device="cuda")
        torch.testing.assert_close(norm, expected_norm, atol=1.0, rtol=1e-5)
        # dst = src * scale = 0.5 (exact for uniform inputs)
        expected_dst = torch.full((numel,), 0.5, device="cuda")
        torch.testing.assert_close(dst, expected_dst)


if __name__ == "__main__":
    unittest.main()
