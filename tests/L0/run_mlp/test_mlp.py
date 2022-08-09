"""Tests for c++ MLP"""
import unittest
from time import time

import torch
from torch import nn

from apex.mlp import MLP


batch_size = 1024
mlp_sizes = [480, 1024, 1024, 512, 256, 1]
num_iters = 10


# note(crcrpar): On Ampere, this test should be run without TF32 enabled.
class TestMLP(unittest.TestCase):
    def test_creation(self):
        MLP(mlp_sizes)

    def test_numeric(self):
        mlp = MLP(mlp_sizes).cuda()

        mlp_layers = []
        for i in range(mlp.num_layers):
            linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
            mlp.weights[i].data.copy_(linear.weight)
            mlp.biases[i].data.copy_(linear.bias)
            mlp_layers.append(linear)
            mlp_layers.append(nn.ReLU(inplace=True))

        ref_mlp = nn.Sequential(*mlp_layers).cuda()

        test_input = (
            torch.empty(batch_size, mlp_sizes[0], device="cuda")
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        ref_input = test_input.clone().detach().requires_grad_()
        mlp_out = mlp(test_input)
        ref_out = ref_mlp(ref_input)
        torch.testing.assert_close(mlp_out, ref_out)

        # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        mlp_out.mean().mul(10.0).backward()
        ref_out.mean().mul(10.0).backward()
        torch.testing.assert_close(test_input.grad, ref_input.grad)
        torch.testing.assert_close(mlp.biases[0].grad, ref_mlp[0].bias.grad)

    def test_no_bias(self):
        for use_activation in ["none", "relu", "sigmoid"]:
            with self.subTest(use_activation=use_activation):
                mlp = MLP(mlp_sizes, bias=False, activation=use_activation).cuda()

                mlp_layers = []
                for i in range(mlp.num_layers):
                    linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1], bias=False)
                    mlp.weights[i].data.copy_(linear.weight)
                    mlp_layers.append(linear)
                    if use_activation == "relu":
                        mlp_layers.append(nn.ReLU(inplace=True))
                    if use_activation == "sigmoid":
                        mlp_layers.append(nn.Sigmoid())

                ref_mlp = nn.Sequential(*mlp_layers).cuda()

                test_input = (
                    torch.empty(batch_size, mlp_sizes[0], device="cuda")
                    .uniform_(-1.0, 1.0)
                    .requires_grad_()
                )
                ref_input = test_input.clone().detach().requires_grad_()
                mlp_out = mlp(test_input)
                ref_out = ref_mlp(ref_input)
                torch.testing.assert_close(mlp_out, ref_out)

                # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
                mlp_out.mean().mul(10.0).backward()
                ref_out.mean().mul(10.0).backward()
                torch.testing.assert_close(test_input.grad, ref_input.grad)
                torch.testing.assert_close(mlp.weights[0].grad, ref_mlp[0].weight.grad)

    def test_with_bias(self):
        for use_activation in ["none", "relu", "sigmoid"]:
            with self.subTest(use_activation=use_activation):
                mlp = MLP(mlp_sizes, bias=True, activation=use_activation).cuda()

                mlp_layers = []
                for i in range(mlp.num_layers):
                    linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1], bias=True)
                    mlp.weights[i].data.copy_(linear.weight)
                    mlp.biases[i].data.copy_(linear.bias)
                    mlp_layers.append(linear)
                    if use_activation == "relu":
                        mlp_layers.append(nn.ReLU(inplace=True))
                    if use_activation == "sigmoid":
                        mlp_layers.append(nn.Sigmoid())

                ref_mlp = nn.Sequential(*mlp_layers).cuda()

                test_input = (
                    torch.empty(batch_size, mlp_sizes[0], device="cuda")
                    .uniform_(-1.0, 1.0)
                    .requires_grad_()
                )
                ref_input = test_input.clone().detach().requires_grad_()
                mlp_out = mlp(test_input)
                ref_out = ref_mlp(ref_input)
                torch.testing.assert_close(mlp_out, ref_out)

                # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
                mlp_out.mean().mul(10.0).backward()
                ref_out.mean().mul(10.0).backward()
                torch.testing.assert_close(test_input.grad, ref_input.grad)
                torch.testing.assert_close(mlp.weights[0].grad, ref_mlp[0].weight.grad)
                torch.testing.assert_close(mlp.biases[0].grad, ref_mlp[0].bias.grad)

    def test_no_grad(self):
        mlp = MLP(mlp_sizes).cuda()

        mlp_layers = []
        for i in range(mlp.num_layers):
            linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
            mlp.weights[i].data.copy_(linear.weight)
            mlp.biases[i].data.copy_(linear.bias)
            mlp_layers.append(linear)
            mlp_layers.append(nn.ReLU(inplace=True))

        ref_mlp = nn.Sequential(*mlp_layers).cuda()

        test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1.0, 1.0)
        ref_input = test_input.clone().detach()
        mlp_out = mlp(test_input)
        ref_out = ref_mlp(ref_input)
        torch.testing.assert_close(mlp_out, ref_out)

        # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        mlp_out.mean().mul(10.0).backward()
        ref_out.mean().mul(10.0).backward()
        torch.testing.assert_close(mlp.weights[0].grad, ref_mlp[0].weight.grad)

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


if __name__ == "__main__":
    unittest.main()
