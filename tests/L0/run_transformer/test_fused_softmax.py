"""Test for fused softmax functions.

Ref: https://github.com/NVIDIA/Megatron-LM/blob/40becfc96c4144985458ac0e0fae45dbb111fbd2/megatron/fused_kernels/tests/test_fused_kernels.py
"""  # NOQA
import itertools

import torch
from torch.testing._internal import common_utils

from apex.transformer import AttnMaskType
from apex.transformer.functional import FusedScaleMaskSoftmax


def attention_mask_func(attention_scores, attention_mask):
    return attention_scores.masked_fill(attention_mask, -10000.0)


autocast_dtypes = (
    (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)
)


class TestFusedScaleMaskSoftmax(common_utils.TestCase):
    def _setup_fused_softmax(
        self,
        input_in_fp16,
        input_in_bf16,
        scale=None,
        softmax_in_fp32=False,
        attn_mask_type=AttnMaskType.padding,
    ):
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
        for (dtype, scale, softmax_in_fp32, shape) in itertools.product(
            (torch.half, torch.bfloat16), (None, 2.0), (False, True), ((4, 12, 24, 24), (32, 12, 4, 214))
        ):
            with self.subTest(f"{dtype}-{scale}-{softmax_in_fp32}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                if not (scale is None or softmax_in_fp32):
                    with self.assertRaises(RuntimeError):
                        self._setup_fused_softmax(
                            input_in_fp16,
                            input_in_bf16,
                            scale,
                            softmax_in_fp32,
                            AttnMaskType.padding,
                        )
                    return
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16,
                    input_in_bf16,
                    scale,
                    softmax_in_fp32,
                    AttnMaskType.padding,
                )

                attention_scores_0 = (
                    torch.randn(shape)
                    .to(device="cuda", dtype=dtype)
                    .requires_grad_(True)
                )
                with torch.no_grad():
                    attention_scores_1 = attention_scores_0.clone().requires_grad_(True)
                mask_shape = (shape[0],) + (1,) + shape[2:]
                mask = torch.randint(0, 2, mask_shape, device="cuda").bool()
                expected = fused_fn(attention_scores_0, mask)
                actual = torch_fn(attention_scores_1, mask)
                self.assertEqual(actual, expected)

                g0 = torch.rand_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                expected.backward(g0)
                actual.backward(g1)

    def test_autocast_fused_scale_mask_softmax(self):
        for dtype in autocast_dtypes:
            with self.subTest(f"{dtype}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, attn_mask_type=AttnMaskType.padding
                )

                attention_scores_0 = (
                    torch.randn((4, 12, 24, 24)).cuda().requires_grad_(True)
                )
                with torch.no_grad():
                    attention_scores_1 = (
                        attention_scores_0.clone().to(dtype).requires_grad_(True)
                    )
                mask = torch.randint(0, 2, (4, 1, 24, 24)).bool().cuda()

                expected = torch_fn(attention_scores_1, mask)
                with torch.cuda.amp.autocast(dtype=dtype):
                    actual = fused_fn(attention_scores_0, mask)
                    self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual, expected)

                g0 = torch.rand_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                expected.backward(g0)
                actual.backward(g1)

    def test_fused_upper_triangle_mask_softmax(self):
        """
        attn_weights.shape: [4, 12, 24, 24]
        total_mask.shape: [4, 1, 24, 24]

        total_mask[0, 0], a 24x24 matrix is like a lower triangular matrix, but
        upper elements are True and lower elements and diagonal are False.
        """
        for (dtype, scale, softmax_in_fp32) in itertools.product(
            (torch.half, torch.bfloat16), (None, 2.0), (False, True),
        ):
            with self.subTest(f"{dtype}-{scale}-{softmax_in_fp32}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                if not (scale is None or softmax_in_fp32):
                    with self.assertRaises(RuntimeError):
                        self._setup_fused_softmax(
                            input_in_fp16,
                            input_in_bf16,
                            scale,
                            softmax_in_fp32,
                            AttnMaskType.causal,
                        )
                    return
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16,
                    input_in_bf16,
                    scale,
                    softmax_in_fp32,
                    AttnMaskType.causal,
                )

                attn_weights_0 = (
                    torch.randn((4, 12, 24, 24))
                    .to(device="cuda", dtype=dtype)
                    .requires_grad_(True)
                )
                with torch.no_grad():
                    attn_weights_1 = attn_weights_0.clone().requires_grad_(True)
                total_mask = (
                    ~(torch.tril(torch.randn((24, 24), device="cuda")).bool())
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                total_mask = total_mask.repeat((4, 1, 1, 1))
                expected = fused_fn(attn_weights_0, total_mask)
                actual = torch_fn(attn_weights_1, total_mask)
                self.assertEqual(actual, expected)

                g0 = torch.randn_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                actual.backward(g0)
                expected.backward(g1)

    def test_autocast_fused_upper_triangle_mask_softmax(self):
        for dtype in autocast_dtypes:
            with self.subTest(f"{dtype}"):
                input_in_fp16 = dtype == torch.half
                input_in_bf16 = dtype == torch.bfloat16
                fused_fn, torch_fn = self._setup_fused_softmax(
                    input_in_fp16, input_in_bf16, attn_mask_type=AttnMaskType.causal
                )

                attn_weights_0 = (
                    torch.randn((4, 12, 24, 24)).cuda().requires_grad_(True)
                )
                with torch.no_grad():
                    attn_weights_1 = (
                        attn_weights_0.clone().to(dtype).requires_grad_(True)
                    )
                total_mask = (
                    ~(torch.tril(torch.randn((24, 24), device="cuda")).bool())
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                with torch.cuda.amp.autocast(dtype=dtype):
                    actual = fused_fn(attn_weights_0, total_mask)
                    self.assertEqual(actual.dtype, dtype)
                expected = torch_fn(attn_weights_1, total_mask)
                self.assertEqual(actual, expected)

                g0 = torch.randn_like(actual)
                with torch.no_grad():
                    g1 = g0.clone()
                actual.backward(g0)
                expected.backward(g1)

if __name__ == "__main__":
    common_utils.run_tests()
