"""Tests for c++ MLP"""
from itertools import product
from time import time

import torch
from torch import nn
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import onlyCUDA

from apex.mlp import MLP


batch_size = 1024
mlp_sizes = [480, 1024, 1024, 512, 256, 1]
num_iters = 10


# note(crcrpar): On Ampere, this test should be run without TF32 enabled.
class TestMLP(common_utils.TestCase):
    def test_creation(self):
        MLP(mlp_sizes)

    def test_numeric(self):
        mlp = MLP(mlp_sizes).cuda()

        mlp_layers = []
        for i in range(mlp.num_layers):
            linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
            with torch.no_grad():
                mlp.weights[i].copy_(linear.weight)
                mlp.biases[i].copy_(linear.bias)
            mlp_layers.append(linear)
            mlp_layers.append(nn.ReLU())

        ref_mlp = nn.Sequential(*mlp_layers).cuda()

        test_input = (
            torch.empty(batch_size, mlp_sizes[0], device="cuda")
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        ref_input = test_input.clone().detach().requires_grad_()
        mlp_out = mlp(test_input)
        ref_out = ref_mlp(ref_input)
        self.assertEqual(mlp_out, ref_out)

        # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        mlp_out.mean().mul(10.0).backward()
        ref_out.mean().mul(10.0).backward()
        self.assertEqual(test_input.grad, ref_input.grad)
        self.assertEqual(mlp.biases[0].grad, ref_mlp[0].bias.grad)

    def _test_mlp_impl(self, use_activation: str, bias: bool, enable_autocast: bool):
        mlp = MLP(mlp_sizes, bias=bias, activation=use_activation).cuda()

        mlp_layers = []
        for i in range(mlp.num_layers):
            linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1], bias=bias)
            with torch.no_grad():
                mlp.weights[i].copy_(linear.weight)
                if bias:
                    mlp.biases[i].copy_(linear.bias)
            mlp_layers.append(linear)
            if use_activation == "relu":
                mlp_layers.append(nn.ReLU())
            if use_activation == "sigmoid":
                mlp_layers.append(nn.Sigmoid())

        ref_mlp = nn.Sequential(*mlp_layers).cuda()

        test_input = (
            torch.empty(batch_size, mlp_sizes[0], device="cuda")
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        ref_input = test_input.clone().detach().requires_grad_()

        with torch.cuda.amp.autocast_mode.autocast(enabled=enable_autocast):
            mlp_out = mlp(test_input)
            mlp_loss = mlp_out.mean().mul(10.0)
            # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
            ref_out = ref_mlp(ref_input)
            ref_loss = ref_out.mean().mul(10.0)

        mlp_loss.backward()
        ref_loss.backward()
        if enable_autocast:
            self.assertEqual(mlp_out.dtype, torch.float16)
            self.assertEqual(ref_out.dtype, torch.float16)
        else:
            self.assertEqual(mlp_out, ref_out)
            self.assertEqual(test_input.grad, ref_input.grad)
            self.assertEqual(mlp.weights[0].grad, ref_mlp[0].weight.grad)

    @common_utils.parametrize(
        "use_activation,bias",
        list(product(("none", "relu", "sigmoid"), (True, False))),
    )
    def test_mlp(self, use_activation: str, bias: bool):
        self._test_mlp_impl(use_activation, bias, enable_autocast=False)

    @common_utils.parametrize(
        "use_activation,bias",
        list(product(("none", "relu", "sigmoid"), (True, False))),
    )
    def test_mlp_autocast_fp16(self, use_activation: str, bias: bool):
        self._test_mlp_impl(use_activation, bias, enable_autocast=True)

    def test_no_grad(self):
        mlp = MLP(mlp_sizes).cuda()

        mlp_layers = []
        for i in range(mlp.num_layers):
            linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
            with torch.no_grad():
                mlp.weights[i].copy_(linear.weight)
                mlp.biases[i].copy_(linear.bias)
            mlp_layers.append(linear)
            mlp_layers.append(nn.ReLU(inplace=True))

        ref_mlp = nn.Sequential(*mlp_layers).cuda()

        test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1.0, 1.0)
        ref_input = test_input.clone().detach()
        mlp_out = mlp(test_input)
        ref_out = ref_mlp(ref_input)
        self.assertEqual(mlp_out, ref_out)

        # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        mlp_out.mean().mul(10.0).backward()
        ref_out.mean().mul(10.0).backward()
        self.assertEqual(mlp.weights[0].grad, ref_mlp[0].weight.grad)

    def test_performance_half(self):
        mlp = MLP(mlp_sizes).cuda().half()

        mlp_layers = []
        for i in range(mlp.num_layers):
            linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
            mlp.weights[i].data.copy_(linear.weight)
            mlp.biases[i].data.copy_(linear.bias)
            mlp_layers.append(linear)
            mlp_layers.append(nn.ReLU(inplace=True))

        ref_mlp = nn.Sequential(*mlp_layers).cuda().half()

        test_input = (
            torch.empty(batch_size, mlp_sizes[0], device="cuda", dtype=torch.half)
            .fill_(10.0)
            .requires_grad_()
        )
        ref_input = (
            torch.empty(batch_size, mlp_sizes[0], device="cuda", dtype=torch.half)
            .fill_(10.0)
            .requires_grad_()
        )

        # Warm up GPU
        for _ in range(100):
            ref_out = ref_mlp(ref_input)
            ref_loss = ref_out.mean()
            ref_mlp.zero_grad()
            ref_loss.backward()
            mlp_out = mlp(test_input)
            test_loss = mlp_out.mean()
            mlp.zero_grad()
            test_loss.backward()

        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            ref_out = ref_mlp(ref_input)
            ref_loss = ref_out.mean()
            ref_mlp.zero_grad()
            ref_loss.backward()
        torch.cuda.synchronize()
        stop_time = time()
        ref_time = (stop_time - start_time) * 1000.0 / num_iters
        print(f"\nPytorch MLP time {ref_time:.4f} ms")

        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            mlp_out = mlp(test_input)
            test_loss = mlp_out.mean()
            mlp.zero_grad()
            test_loss.backward()
        torch.cuda.synchronize()
        stop_time = time()
        actual_time = (stop_time - start_time) * 1000.0 / num_iters
        print(f"C++ MLP time {actual_time:.4f} ms")
        torch.cuda.profiler.stop()
        self.assertLessEqual(
            actual_time,
            ref_time,
            msg=f"Custom extension took {actual_time:.4f} while PyTorch took {ref_time:.4f}",
        )


instantiate_device_type_tests(TestMLP, globals(), only_for=("cuda",))


if __name__ == "__main__":
    common_utils.run_tests()
