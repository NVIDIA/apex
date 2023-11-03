#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are not permit-
# ted.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import unittest

SKIP_TEST = None
try:
    from apex.contrib.group_norm.group_norm import cuda_group_norm_nhwc_one_pass
    from apex.contrib.group_norm.group_norm import cuda_group_norm_nhwc_two_pass
    from apex.contrib.group_norm.group_norm import torch_group_norm
    from apex.contrib.group_norm import GroupNorm
except ImportError as e:
    SKIP_TEST = e


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class GroupNormTest(unittest.TestCase):

    def setUp(self, seed=0):
        super().setUp()
        torch.manual_seed(seed)

    def verify_group_norm(self,
                          tst_func,
                          N=32,
                          C=128,
                          H=256,
                          W=256,
                          G=32,
                          ref_func=torch_group_norm,
                          xdtype=torch.float16,
                          wdtype=torch.float32,
                          eps=1e-5,
                          memory_format=torch.channels_last,
                          device='cuda',
                          act=""):
        # create data
        x_shape = (N, C, H, W)
        w_shape = (C,)
        weight = torch.rand(w_shape,
                            dtype=wdtype,
                            device='cuda',
                            requires_grad=True)
        bias = torch.rand(w_shape,
                          dtype=wdtype,
                          device='cuda',
                          requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=xdtype, device='cuda')
        x = x.to(memory_format=memory_format)
        dy = .1 * torch.randn_like(x)
        x.requires_grad_(True)

        # forward pass
        y_ref = ref_func(x, G, weight, bias, eps, act)
        if tst_func is GroupNorm:
            gn = GroupNorm(G, C, eps, device=device, dtype=wdtype, act=act)
            with torch.no_grad():
                gn.weight = torch.nn.Parameter(weight)
                gn.bias = torch.nn.Parameter(bias)
            y_tst = gn(x)
        else:
            y_tst = tst_func(x, G, weight, bias, eps, act)

        # backward pass
        y_ref.backward(dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [t.grad.clone() for t in [x, weight, bias]]
        x.grad.zero_()
        weight.grad.zero_()
        bias.grad.zero_()
        y_tst.backward(dy, retain_graph=True)
        if tst_func is GroupNorm:
            dx_tst, dw_tst, db_tst = x.grad, gn.weight.grad, gn.bias.grad
        else:
            dx_tst, dw_tst, db_tst = [t.grad.clone() for t in [x, weight, bias]]

        # compare
        torch.testing.assert_close(y_tst, y_ref, atol=4e-2, rtol=0)
        torch.testing.assert_close(dx_tst, dx_ref, atol=4e-2, rtol=0)
        torch.testing.assert_close(dw_tst, dw_ref, atol=4e-2, rtol=0)
        torch.testing.assert_close(db_tst, db_ref, atol=4e-2, rtol=0)

    def test_fp16_one_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, act="")

    def test_fp16_two_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, act="")

    def test_fp16_one_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, act="swish")

    def test_fp16_two_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, act="swish")

    def test_bf16_one_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass,
                               xdtype=torch.bfloat16,
                               act="")

    def test_bf16_two_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass,
                               xdtype=torch.bfloat16,
                               act="")

    def test_bf16_one_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass,
                               xdtype=torch.bfloat16,
                               act="swish")

    def test_bf16_two_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass,
                               xdtype=torch.bfloat16,
                               act="swish")

    def test_fp32_one_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass,
                               xdtype=torch.float32,
                               act="")

    def test_fp32_two_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass,
                               xdtype=torch.float32,
                               act="")

    def test_fp32_one_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass,
                               xdtype=torch.float32,
                               act="swish")

    def test_fp32_two_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass,
                               xdtype=torch.float32,
                               act="swish")

    def test_group_norm_module(self):
        self.verify_group_norm(GroupNorm, G=16, act="swish")

    def test_16_groups(self):
        sizes = [
            [8, 2560, 16, 16],
            [8, 1920, 32, 32],
            [8, 1920, 16, 16],
            [8, 2560, 8, 8],
        ]
        for sz in sizes:
            n, c, h, w = sz
            self.verify_group_norm(GroupNorm,
                                   N=n,
                                   C=c,
                                   H=h,
                                   W=w,
                                   G=16,
                                   act="swish")

    def test_fp16_parameters(self):
        n, c, h, w = 8, 2560, 16, 16
        self.verify_group_norm(GroupNorm,
                               N=n,
                               C=c,
                               H=h,
                               W=w,
                               G=16,
                               xdtype=torch.float16,
                               wdtype=torch.float16,
                               act="swish")


if __name__ == '__main__':
    unittest.main()
