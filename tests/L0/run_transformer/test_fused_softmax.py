"""Test for fused softmax functions.

Ref: https://github.com/NVIDIA/Megatron-LM/blob/40becfc96c4144985458ac0e0fae45dbb111fbd2/megatron/fused_kernels/tests/test_fused_kernels.py
"""  # NOQA
import itertools
import unittest

import torch

from apex.transformer import AttnMaskType
from apex.transformer.functional import FusedScaleMaskSoftmax


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


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
        """
        attention_scores.shape = [4, 12, 24, 24]
        mask.shape = [4, 1, 24, 24]
        """
        for (dtype, scale, softmax_in_fp32) in itertools.product(
                (torch.half, torch.bfloat16),
                (None, 2.0),
                (False, True),
        ):
            with self.subTest(f"{dtype}-{scale}-{softmax_in_fp32}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                if not (scale is None or softmax_in_fp32):
                    with self.assertRaises(RuntimeError):
                        self._setup_fused_softmax(input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.padding)
                    return
                fused_fn, torch_fn = self._setup_fused_softmax(input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.padding)

                attention_scores = torch.randn((4, 12, 24, 24)).to(device="cuda", dtype=dtype)
                mask = torch.randint(0, 2, (4, 1, 24, 24), device="cuda").bool()
                reference = fused_fn(attention_scores, mask)
                actual = torch_fn(attention_scores, mask)
                torch.testing.assert_allclose(actual, reference)

    def test_autocast_fused_scale_mask_softmax(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(f"{dtype}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, attn_mask_type=AttnMaskType.padding)
                attention_scores = torch.randn((4, 12, 24, 24)).cuda()
                mask = torch.randint(0, 2, (4, 1, 24, 24)).bool().cuda()

                with torch.cuda.amp.autocast(dtype=dtype):
                    actual = fused_fn(attention_scores, mask)
                    self.assertEqual(actual.dtype, dtype)
                with torch.no_grad():
                    expected = torch_fn(attention_scores.to(dtype), mask)
                torch.testing.assert_allclose(actual, expected)

    def test_fused_upper_triangle_mask_softmax(self):
        """
        attn_weights.shape: [4, 12, 24, 24]
        total_mask.shape: [4, 1, 24, 24]

        total_mask[0, 0], a 24x24 matrix is like a lower triangular matrix, but
        upper elements are True and lower elements and diagonal are False.
        """
        for (dtype, scale, softmax_in_fp32) in itertools.product(
                (torch.half, torch.bfloat16),
                (None, 2.0),
                (False, True),
        ):
            with self.subTest(f"{dtype}-{scale}-{softmax_in_fp32}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                if not (scale is None or softmax_in_fp32):
                    with self.assertRaises(RuntimeError):
                        self._setup_fused_softmax(
                            input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.causal)
                    return
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, scale, softmax_in_fp32, AttnMaskType.causal)

                attn_weights = torch.randn((4, 12, 24, 24)).to(device="cuda", dtype=dtype)
                total_mask = (~(
                    torch.tril(torch.randn((24, 24), device="cuda")).bool()
                ).unsqueeze(0).unsqueeze(0))
                total_mask = total_mask.repeat((4, 1, 1, 1))
                reference = fused_fn(attn_weights, total_mask)
                actual = torch_fn(attn_weights, total_mask)
                torch.testing.assert_allclose(actual, reference)

    def test_autocast_fused_upper_triangle_mask_softmax(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(f"{dtype}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, attn_mask_type=AttnMaskType.causal)
                attn_weights = torch.randn((4, 12, 24, 24)).cuda()
                total_mask = (~(
                    torch.tril(torch.randn((24, 24), device="cuda")).bool()
                ).unsqueeze(0).unsqueeze(0))

                with torch.cuda.amp.autocast(dtype=dtype):
                    actual = fused_fn(attn_weights, total_mask)
                    self.assertEqual(actual.dtype, dtype)
                with torch.no_grad():
                    expected = torch_fn(attn_weights.to(dtype), total_mask)
                torch.testing.assert_allclose(actual, expected)
