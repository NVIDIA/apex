"""Tests for c++ MLP"""
import unittest
from time import time
import numpy as np

import torch
from torch import nn

from apex.mlp import MLP

batch_size = 1024
mlp_sizes = [480, 1024, 1024, 512, 256, 1]
num_iters = 10

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

        test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
        ref_input = test_input.clone().detach().requires_grad_()
        mlp_out = mlp(test_input)
        ref_out = ref_mlp(ref_input)
        np.testing.assert_allclose(
            mlp_out.detach().cpu().numpy(),
            ref_out.detach().cpu().numpy(),
            atol=1e-7, rtol=1e-5)

        # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        mlp_out.mean().mul(10.).backward()
        ref_out.mean().mul(10.).backward()
        np.testing.assert_allclose(
            test_input.grad.detach().cpu().numpy(),
            ref_input.grad.detach().cpu().numpy(),
            atol=0, rtol=1e-5)
        np.testing.assert_allclose(
            mlp.biases[0].grad.detach().cpu().numpy(),
            ref_mlp[0].bias.grad.detach().cpu().numpy(),
            atol=1e-7, rtol=1e-5)

    def test_no_bias(self):
        for use_activation in ['none', 'relu', 'sigmoid']:
            mlp = MLP(mlp_sizes, bias=False, activation=use_activation).cuda()

            mlp_layers = []
            for i in range(mlp.num_layers):
                linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1], bias=False)
                mlp.weights[i].data.copy_(linear.weight)
                mlp_layers.append(linear)
                if use_activation == 'relu':
                    mlp_layers.append(nn.ReLU(inplace=True))
                if use_activation == 'sigmoid':
                    mlp_layers.append(nn.Sigmoid())

            ref_mlp = nn.Sequential(*mlp_layers).cuda()

            test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
            ref_input = test_input.clone().detach().requires_grad_()
            mlp_out = mlp(test_input)
            ref_out = ref_mlp(ref_input)
            np.testing.assert_allclose(
                mlp_out.detach().cpu().numpy(),
                ref_out.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

            # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
            mlp_out.mean().mul(10.).backward()
            ref_out.mean().mul(10.).backward()
            np.testing.assert_allclose(
                test_input.grad.detach().cpu().numpy(),
                ref_input.grad.detach().cpu().numpy(),
                atol=0, rtol=100)
            np.testing.assert_allclose(
                mlp.weights[0].grad.detach().cpu().numpy(),
                ref_mlp[0].weight.grad.detach().cpu().numpy(),
                atol=1e-7, rtol=100)

    def test_with_bias(self):
        for use_activation in ['none', 'relu', 'sigmoid']:
            mlp = MLP(mlp_sizes, bias=True, activation=use_activation).cuda()

            mlp_layers = []
            for i in range(mlp.num_layers):
                linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1], bias=True)
                mlp.weights[i].data.copy_(linear.weight)
                mlp.biases[i].data.copy_(linear.bias)
                mlp_layers.append(linear)
                if use_activation == 'relu':
                    mlp_layers.append(nn.ReLU(inplace=True))
                if use_activation == 'sigmoid':
                    mlp_layers.append(nn.Sigmoid())

            ref_mlp = nn.Sequential(*mlp_layers).cuda()

            test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
            ref_input = test_input.clone().detach().requires_grad_()
            mlp_out = mlp(test_input)
            ref_out = ref_mlp(ref_input)
            np.testing.assert_allclose(
                mlp_out.detach().cpu().numpy(),
                ref_out.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

            # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
            mlp_out.mean().mul(10.).backward()
            ref_out.mean().mul(10.).backward()
            np.testing.assert_allclose(
                test_input.grad.detach().cpu().numpy(),
                ref_input.grad.detach().cpu().numpy(),
                atol=0, rtol=1)
            np.testing.assert_allclose(
                mlp.weights[0].grad.detach().cpu().numpy(),
                ref_mlp[0].weight.grad.detach().cpu().numpy(),
                atol=1e-7, rtol=1)
            np.testing.assert_allclose(
                mlp.biases[0].grad.detach().cpu().numpy(),
                ref_mlp[0].bias.grad.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

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

        test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.)
        ref_input = test_input.clone().detach()
        mlp_out = mlp(test_input)
        ref_out = ref_mlp(ref_input)
        np.testing.assert_allclose(
            mlp_out.detach().cpu().numpy(),
            ref_out.detach().cpu().numpy(),
            atol=1e-7, rtol=1e-5)

        # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        mlp_out.mean().mul(10.).backward()
        ref_out.mean().mul(10.).backward()
        np.testing.assert_allclose(
            mlp.weights[0].grad.detach().cpu().numpy(),
            ref_mlp[0].weight.grad.detach().cpu().numpy(),
            atol=1e-7, rtol=1e-5)


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

        test_input = torch.empty(
            batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()
        ref_input = torch.empty(
            batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()

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
        print(F"\nPytorch MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            mlp_out = mlp(test_input)
            test_loss = mlp_out.mean()
            mlp.zero_grad()
            test_loss.backward()
        torch.cuda.synchronize()
        stop_time = time()
        print(F"C++ MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
        torch.cuda.profiler.stop()

if __name__ == '__main__':
    unittest.main()
