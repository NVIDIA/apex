import unittest

import torch

from apex.contrib.optimizers.fused_lamb import FusedLAMB as DeprecatedFusedLAMB
from apex.optimizers import FusedAdagrad, FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD


_OPTIMIZERS = (FusedAdagrad, FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD, DeprecatedFusedLAMB)


def _zero_grad_optimizer(optimizer_class, params, set_grad_none):
    # zero_grad is Python-only. Initialize the real Optimizer base without the
    # subclass's CUDA-kernel setup, which is unrelated to gradient clearing.
    optimizer = optimizer_class.__new__(optimizer_class)
    torch.optim.Optimizer.__init__(optimizer, params, {})
    optimizer.set_grad_none = set_grad_none
    return optimizer


class TestZeroGradConfiguration(unittest.TestCase):
    def test_preserves_and_zeros_existing_gradient_buffers(self):
        for optimizer_class in _OPTIMIZERS:
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(optimizer=optimizer_class.__module__, dtype=dtype):
                    first = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype))
                    second = torch.nn.Parameter(torch.ones(3, dtype=dtype))
                    unused = torch.nn.Parameter(torch.ones(1, dtype=dtype))
                    first.grad = torch.full_like(first, 3, requires_grad=True)
                    # A view gradient exercises the base optimizer's detach path.
                    second.grad = torch.arange(6, dtype=dtype)[::2]
                    original_first = first.grad
                    original_second = second.grad
                    optimizer = _zero_grad_optimizer(
                        optimizer_class, [{"params": [first]}, {"params": [second, unused]}], False
                    )
                    optimizer.zero_grad()
                    self.assertIs(first.grad, original_first)
                    self.assertIs(second.grad, original_second)
                    torch.testing.assert_close(first.grad, torch.zeros_like(first), rtol=0, atol=0)
                    torch.testing.assert_close(
                        second.grad, torch.zeros_like(second), rtol=0, atol=0
                    )
                    self.assertFalse(first.grad.requires_grad)
                    self.assertIsNone(unused.grad)
                    # A repeated clear does not replace buffers or touch parameters.
                    optimizer.zero_grad()
                    self.assertIs(first.grad, original_first)
                    torch.testing.assert_close(first, torch.ones_like(first), rtol=0, atol=0)
                    torch.testing.assert_close(second, torch.ones_like(second), rtol=0, atol=0)

    def test_set_grad_none_true_still_clears_to_none(self):
        for optimizer_class in _OPTIMIZERS:
            with self.subTest(optimizer=optimizer_class.__module__):
                param = torch.nn.Parameter(torch.ones(3))
                param.grad = torch.ones_like(param)
                optimizer = _zero_grad_optimizer(optimizer_class, [param], True)
                optimizer.zero_grad()
                self.assertIsNone(param.grad)
                optimizer.zero_grad()
                self.assertIsNone(param.grad)

    def test_zero_gradient_keeps_weight_decay_step(self):
        for optimizer_class in _OPTIMIZERS:
            with self.subTest(optimizer=optimizer_class.__module__):
                param = torch.nn.Parameter(torch.ones(3))
                param.grad = torch.ones_like(param)
                optimizer = _zero_grad_optimizer(optimizer_class, [param], False)
                optimizer.zero_grad()
                # Use an ordinary CPU optimizer to demonstrate the observable
                # zero-versus-None distinction without claiming a CUDA update.
                updater = torch.optim.SGD([param], lr=0.1, weight_decay=0.2)
                updater.step()
                torch.testing.assert_close(param, torch.full_like(param, 0.98), rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
