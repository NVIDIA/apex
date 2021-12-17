"""Test for fused softmax functions.

Ref: https://github.com/NVIDIA/Megatron-LM/blob/40becfc96c4144985458ac0e0fae45dbb111fbd2/megatron/fused_kernels/tests/test_fused_kernels.py
"""  # NOQA
import itertools
import unittest

import torch

from apex.transformer import AttnMaskType
from apex.transformer.functional import FusedScaleMaskSoftmax


def attention_mask_func(attention_scores, attention_mask):
    return attention_scores.masked_fill(attention_mask, -10000.0)


autocast_dtypes = (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)
# In general, input is a 4D tensor and mask is also a 4D tensor.
# The fused kernels assume input to have the size of (b, np, sq, sk) and mask to have (b, 1, sq, sk).
input_shapes = [(4, 12, 24, 24), (4, 13, 8, 25)]

class TestFusedScaleMaskSoftmax(unittest.TestCase):

    def _setup_fused_softmax(self, input_in_fp16, input_in_bf16, scale=None, softmax_in_fp32=False, attn_mask_type=AttnMaskType.padding):
        fused_fn = FusedScaleMaskSoftmax(
            input_in_fp16=input_in_fp16,
            input_in_bf16=input_in_bf16,
            mask_func=attention_mask_func,
            scale=scale,
            softmax_in_fp32=softmax_in_fp32,
            attn_mask_type=attn_mask_type,
            scaled_masked_softmax_fusion=True,
        )
        torch_fn = FusedScaleMaskSoftmax(
            input_in_fp16=input_in_fp16,
            input_in_bf16=input_in_bf16,
            mask_func=attention_mask_func,
            scale=scale,
            softmax_in_fp32=softmax_in_fp32,
            attn_mask_type=attn_mask_type,
            scaled_masked_softmax_fusion=False,
        )
        return fused_fn, torch_fn

    def test_fused_scale_mask_softmax(self):
        for (dtype, scale, softmax_in_fp32, input_shape) in itertools.product(
                (torch.half, torch.bfloat16),
                (None, 2.0),
                (False, True),
                input_shapes
        ):
            with self.subTest(dtype=dtype, scale=scale, softmax_in_fp32=softmax_in_fp32, input_shape=input_shape):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                if not (scale is None or softmax_in_fp32):
                    with self.assertRaises(RuntimeError):
                        self._setup_fused_softmax(input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.padding)
                    return
                fused_fn, torch_fn = self._setup_fused_softmax(input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.padding)

                attention_scores_0 = torch.randn(input_shape).to(device="cuda", dtype=dtype).requires_grad_(True)
                with torch.no_grad():
                    attention_scores_1 = attention_scores_0.clone().requires_grad_(True)
                mask = torch.randint(0, 2, (input_shape[0], 1) + input_shape[-2:]).to(device="cuda", dtype=torch.bool)
                expected = fused_fn(attention_scores_0, mask)
                actual = torch_fn(attention_scores_1, mask)
                torch.testing.assert_allclose(actual, expected)

                g0 = torch.rand_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                expected.backward(g0)
                actual.backward(g1)

    def test_autocast_fused_scale_mask_softmax(self):
        for dtype, input_shape in itertools.product(autocast_dtypes, input_shapes):
            with self.subTest(dtype=dtype, input_shape=input_shape):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, attn_mask_type=AttnMaskType.padding)

                attention_scores_0 = torch.randn(input_shape).cuda().requires_grad_(True)
                with torch.no_grad():
                    attention_scores_1 = attention_scores_0.clone().to(dtype).requires_grad_(True)
                mask = torch.randint(0, 2, (input_shape[0], 1) + input_shape[-2:]).to(device="cuda", dtype=torch.bool)

                expected = torch_fn(attention_scores_1, mask)
                with torch.cuda.amp.autocast(dtype=dtype):
                    actual = fused_fn(attention_scores_0, mask)
                    self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_allclose(actual, expected)

                g0 = torch.rand_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                expected.backward(g0)
                actual.backward(g1)

    def test_fused_upper_triangle_mask_softmax(self):
        for (dtype, scale, softmax_in_fp32, input_shape) in itertools.product(
                (torch.half, torch.bfloat16),
                (None, 2.0),
                (False, True),
                input_shapes,
        ):
            with self.subTest(dtype=dtype, scale=scale, softmax_in_fp32=softmax_in_fp32, input_shape=input_shape):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                if not (scale is None or softmax_in_fp32):
                    with self.assertRaises(RuntimeError):
                        self._setup_fused_softmax(
                            input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.causal)
                    return
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.causal)

                attn_weights_0 = torch.randn(input_shape).to(device="cuda", dtype=dtype).requires_grad_(True)
                with torch.no_grad():
                    attn_weights_1 = attn_weights_0.clone().requires_grad_(True)
                total_mask = (~(
                    torch.tril(torch.randn(input_shape[-2:], device="cuda")).bool()
                ).unsqueeze(0).unsqueeze(0))
                total_mask = total_mask.repeat((input_shape[0], 1, 1, 1))
                expected = fused_fn(attn_weights_0, total_mask)
                actual = torch_fn(attn_weights_1, total_mask)
                torch.testing.assert_allclose(actual, expected)

                g0 = torch.randn_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                actual.backward(g0)
                expected.backward(g1)

    def test_autocast_fused_upper_triangle_mask_softmax(self):
        for dtype, input_shape in itertools.product(autocast_dtypes, input_shapes):
            with self.subTest(dtype=dtype, input_shape=input_shape):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, attn_mask_type=AttnMaskType.causal)

                attn_weights_0 = torch.randn(input_shape).cuda().requires_grad_(True)
                with torch.no_grad():
                    attn_weights_1 = attn_weights_0.clone().to(dtype).requires_grad_(True)
                total_mask = (~(
                    torch.tril(torch.randn(input_shape[-2:], device="cuda")).bool()
                ).unsqueeze(0).unsqueeze(0))
                total_mask = total_mask.repeat((input_shape[0], 1, 1, 1))

                with torch.cuda.amp.autocast(dtype=dtype):
                    actual = fused_fn(attn_weights_0, total_mask)
                    self.assertEqual(actual.dtype, dtype)
                expected = torch_fn(attn_weights_1, total_mask)
                torch.testing.assert_allclose(actual, expected)

                g0 = torch.randn_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                actual.backward(g0)
                expected.backward(g1)
