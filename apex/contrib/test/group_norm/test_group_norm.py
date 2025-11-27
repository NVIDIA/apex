#!/usr/bin/env python
# coding: utf-8

#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

import functools
import importlib
import pathlib
import sys
import torch
import unittest

SKIP_TEST = None
try:
    from apex.contrib.group_norm.group_norm import cuda_group_norm_nhwc_one_pass
    from apex.contrib.group_norm.group_norm import cuda_group_norm_nhwc_two_pass
    from apex.contrib.group_norm.group_norm import cuda_group_norm_v2_nhwc
    from apex.contrib.group_norm.group_norm import get_cc_and_sm_count
    from apex.contrib.group_norm import GroupNorm
except ImportError as e:
    SKIP_TEST = e


def torch_group_norm_high_precision(x, g, w, b, eps, act="", *, compute_type):
    xdtype = x.dtype
    y = torch.nn.functional.group_norm(
        x.to(compute_type),
        g,
        w.to(compute_type),
        b.to(compute_type),
        eps,
    )
    if act in ["silu", "swish"]:
        y = torch.nn.functional.silu(y)
    y = y.to(dtype=xdtype)
    return y


torch_group_norm_high_precision_fp64 = functools.partial(
    torch_group_norm_high_precision,
    compute_type=torch.float64,
)


@functools.cache
def relative_ulp(dtype, device):
    # Unit in the Last Place
    one = torch.tensor(1.0, dtype=dtype, device=device)
    two = torch.tensor(2.0, dtype=dtype, device=device)
    return (torch.nextafter(one, two) - one).item()


@unittest.skipIf(
    torch.cuda.get_device_properties().multi_processor_count < 16,
    "GroupNorm is unsupported on low SM count devices",
)
@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class GroupNormTest(unittest.TestCase):
    def setUp(self, seed=0):
        super().setUp()
        torch.manual_seed(seed)

    def verify_group_norm(
        self,
        tst_func,
        N=32,
        C=128,
        H=256,
        W=256,
        G=32,
        ref_func=torch_group_norm_high_precision_fp64,
        xdtype=torch.float16,
        wdtype=torch.float32,
        eps=1e-5,
        memory_format=torch.channels_last,
        device="cuda",
        act="",
    ):
        # create data
        x_shape = (N, C, H, W)
        w_shape = (C,)
        weight = torch.rand(w_shape, dtype=wdtype, device="cuda", requires_grad=True)
        bias = torch.rand(w_shape, dtype=wdtype, device="cuda", requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=xdtype, device="cuda")
        x = x.to(memory_format=memory_format)
        dy = 0.1 * torch.randn_like(x)
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
        torch.testing.assert_close(
            y_tst, y_ref, atol=1e-2, rtol=relative_ulp(y_ref.dtype, y_ref.device)
        )
        torch.testing.assert_close(
            dx_tst, dx_ref, atol=1e-2, rtol=relative_ulp(dx_ref.dtype, dx_ref.device)
        )
        torch.testing.assert_close(
            dw_tst, dw_ref, atol=1e-2, rtol=relative_ulp(dw_ref.dtype, dw_ref.device)
        )
        torch.testing.assert_close(
            db_tst, db_ref, atol=1e-2, rtol=relative_ulp(db_ref.dtype, db_ref.device)
        )

    def test_fp16_one_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, act="")

    def test_fp16_two_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, act="")

    def test_fp16_one_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, act="swish")

    def test_fp16_two_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, act="swish")

    def test_bf16_one_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, xdtype=torch.bfloat16, act="")

    def test_bf16_two_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, xdtype=torch.bfloat16, act="")

    def test_bf16_one_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, xdtype=torch.bfloat16, act="swish")

    def test_bf16_two_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, xdtype=torch.bfloat16, act="swish")

    def test_fp32_one_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, xdtype=torch.float32, act="")

    def test_fp32_two_pass_algo(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, xdtype=torch.float32, act="")

    def test_fp32_one_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_one_pass, xdtype=torch.float32, act="swish")

    def test_fp32_two_pass_algo_with_swish(self):
        self.verify_group_norm(cuda_group_norm_nhwc_two_pass, xdtype=torch.float32, act="swish")

    def test_group_norm_module(self):
        self.verify_group_norm(GroupNorm, G=16, act="swish")

    def test_group_norm_inductor(self):
        N, C, H, W, G = 32, 320, 256, 256, 16

        model = (
            torch.nn.Sequential(
                GroupNorm(G, C, act="silu", dtype=torch.float16),
                torch.nn.Conv2d(C, C, kernel_size=3, padding="same"),
            )
            .cuda()
            .half()
        )
        compiled = torch.compile(model)

        x = -2.3 + 0.5 * torch.randn((N, C, H, W), dtype=torch.float16, device="cuda")
        x = x.to(memory_format=torch.channels_last)
        dy = 0.1 * torch.randn_like(x)
        x.requires_grad_(True)

        for _ in range(4):
            y = compiled(x)
            y.backward(dy)

        from torch._dynamo.utils import counters

        # TODO: Remove this when 3.9 is no longer supported
        if sys.version_info < (3, 10):
            num_graph_breaks = sum(counters["graph_break"].values())
        else:
            num_graph_breaks = counters["graph_break"].total()
        self.assertEqual(num_graph_breaks, 0, "Shouldn't see any graph breaks.")
        self.assertEqual(counters["stats"]["unique_graphs"], 1, "Expect only one graph.")

    def test_16_groups(self):
        sizes = [
            [8, 2560, 16, 16],
            [8, 1920, 32, 32],
            [8, 1920, 16, 16],
            [8, 2560, 8, 8],
            [1, 128, 16128, 1200],
        ]
        for sz in sizes:
            n, c, h, w = sz
            self.verify_group_norm(GroupNorm, N=n, C=c, H=h, W=w, G=16, act="swish")

    def test_fp16_parameters(self):
        n, c, h, w = 8, 2560, 16, 16
        self.verify_group_norm(
            GroupNorm,
            N=n,
            C=c,
            H=h,
            W=w,
            G=16,
            xdtype=torch.float16,
            wdtype=torch.float16,
            act="swish",
        )

    @staticmethod
    @functools.cache
    def get_v2_hw_c_list():
        srcpath = pathlib.Path(__file__).parent.absolute()
        gen_module_path = (
            srcpath / ".." / ".." / "csrc" / "group_norm_v2" / "generate_gn_cuda_inst.py"
        )
        spec = importlib.util.spec_from_file_location("generate_gn_cuda_inst", gen_module_path)
        generate_gn_cuda_inst = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_gn_cuda_inst)
        return generate_gn_cuda_inst.hw_c_list

    def check_v2_cc_and_sm_count(self):
        cc, sm_count = get_cc_and_sm_count(torch.cuda.current_device())
        return (
            cc in GroupNorm.GN_V2_SUPPORTED_LOWER_BOUND_SM_COUNT
            and sm_count >= GroupNorm.GN_V2_SUPPORTED_LOWER_BOUND_SM_COUNT[cc]
        )

    def skip_if_v2_not_supported(self):
        if not self.check_v2_cc_and_sm_count():
            cc, sm_count = get_cc_and_sm_count(torch.cuda.current_device())
            self.skipTest(
                f"SM count {sm_count} is not supported for compute capability {cc[0]}.{cc[1]}"
            )

    def test_check_v2_legality(self):
        gn = GroupNorm(
            num_groups=16,
            num_channels=640,
            device="cuda",
            dtype=torch.float16,
            act="swish",
        )
        self.skip_if_v2_not_supported()
        # Correct
        x = torch.empty(
            8,
            640,
            32,
            32,
            dtype=torch.float16,
            device="cuda",
            memory_format=torch.channels_last,
        )
        self.assertTrue(gn._check_legality(x) and gn._check_v2_legality(x))
        # Wrong layout
        x = torch.empty(8, 640, 32, 32, dtype=torch.float16, device="cuda")
        self.assertFalse(gn._check_legality(x) and gn._check_v2_legality(x))
        # Wrong shape
        x = torch.empty(
            8,
            640,
            32,
            24,
            dtype=torch.float16,
            device="cuda",
            memory_format=torch.channels_last,
        )
        self.assertFalse(gn._check_legality(x) and gn._check_v2_legality(x))
        # Wrong dtype
        x = torch.empty(
            8,
            640,
            32,
            32,
            dtype=torch.float32,
            device="cuda",
            memory_format=torch.channels_last,
        )
        self.assertFalse(gn._check_legality(x) and gn._check_v2_legality(x))

    def test_fp16_v2_32_groups(self):
        self.skip_if_v2_not_supported()
        for n in [1, 2, 4, 8, 16, 32]:
            for hw, c in self.get_v2_hw_c_list():
                h = w = int(hw**0.5)
                assert hw == h * w
                self.verify_group_norm(
                    cuda_group_norm_v2_nhwc,
                    N=n,
                    C=c,
                    H=h,
                    W=w,
                    G=32,
                    xdtype=torch.float16,
                    wdtype=torch.float16,
                    act="",
                )

    def test_fp16_v2_16_groups_with_swish(self):
        self.skip_if_v2_not_supported()
        for n in [1, 2, 4, 8, 16, 32]:
            for hw, c in self.get_v2_hw_c_list():
                h = w = int(hw**0.5)
                assert hw == h * w
                self.verify_group_norm(
                    cuda_group_norm_v2_nhwc,
                    N=n,
                    C=c,
                    H=h,
                    W=w,
                    G=16,
                    xdtype=torch.float16,
                    wdtype=torch.float16,
                    act="swish",
                )

    def test_bf16_v2_32_groups(self):
        self.skip_if_v2_not_supported()
        for n in [1, 2, 4, 8, 16, 32]:
            for hw, c in self.get_v2_hw_c_list():
                h = w = int(hw**0.5)
                assert hw == h * w
                self.verify_group_norm(
                    cuda_group_norm_v2_nhwc,
                    N=n,
                    C=c,
                    H=h,
                    W=w,
                    G=32,
                    xdtype=torch.bfloat16,
                    wdtype=torch.bfloat16,
                    act="",
                )

    def test_bf16_v2_16_groups_with_swish(self):
        self.skip_if_v2_not_supported()
        for n in [1, 2, 4, 8, 16, 32]:
            for hw, c in self.get_v2_hw_c_list():
                h = w = int(hw**0.5)
                assert hw == h * w
                self.verify_group_norm(
                    cuda_group_norm_v2_nhwc,
                    N=n,
                    C=c,
                    H=h,
                    W=w,
                    G=16,
                    xdtype=torch.bfloat16,
                    wdtype=torch.bfloat16,
                    act="swish",
                )


if __name__ == "__main__":
    unittest.main()
