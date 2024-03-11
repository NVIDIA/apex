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

def forward_torch_softmax(input, mask, scale):
    input = input * scale
    mask_output = attention_mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    all_k_masked = mask.all(axis=-1)
    zero_attention_mask = (1.0 - all_k_masked.float())[:, :, :, None]
    probs = probs * zero_attention_mask
    return probs

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

    def tearDown(self) -> None:
        torch.cuda.empty_cache()
        super().tearDown()

    def test_fused_scale_mask_softmax(self):
        """
        attention_scores.shape = [4, 12, 24, 24]
        mask.shape = [4, 1, 24, 24]
        """
        for (dtype, scale, softmax_in_fp32, shape) in itertools.product(
            (torch.half, torch.bfloat16), (None, 2.0), (False, True), ((4, 12, 24, 24), (32, 12, 4, 214))
        ):
            msg = f"{dtype}-{scale}-{softmax_in_fp32}"
            input_in_fp16 = dtype == torch.half
            input_in_bf16 = dtype == torch.bfloat16
            if not (scale is None or softmax_in_fp32):
                with self.assertRaises(RuntimeError, msg=msg):
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
            self.assertEqual(actual, expected, msg=msg)

            g0 = torch.rand_like(actual)
            with torch.no_grad():
                g1 = g0.clone()
            expected.backward(g0)
            actual.backward(g1)

    def test_autocast_fused_scale_mask_softmax(self):
        for dtype in autocast_dtypes:
            msg = f"dtype: {dtype}"
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
                self.assertEqual(actual.dtype, dtype, msg=msg)
            self.assertEqual(actual, expected, msg=msg)

            g0 = torch.rand_like(actual)
            with torch.no_grad():
                g1 = g0.clone()
            expected.backward(g0)
            actual.backward(g1)

    def test_fused_scale_softmax(self):
        """
        attention_scores.shape = [4, 12, 24, 24]
        mask = None
        """
        for (dtype, scale, softmax_in_fp32, shape) in itertools.product(
            (torch.half, torch.bfloat16), (None, 2.0), (False, True), ((4, 12, 24, 24), (32, 12, 4, 214))
        ):
            msg = f"{dtype}-{scale}-{softmax_in_fp32}"
            input_in_fp16 = dtype == torch.half
            input_in_bf16 = dtype == torch.bfloat16
            if not (scale is None or softmax_in_fp32):
                with self.assertRaises(RuntimeError, msg=msg):
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
            mask = None

            expected = fused_fn(attention_scores_0, mask)
            actual = torch_fn(attention_scores_1, mask)
            self.assertEqual(actual, expected, msg=msg)

            g0 = torch.rand_like(actual)
            with torch.no_grad():
                g1 = g0.clone()
            expected.backward(g0)
            actual.backward(g1)

    def test_autocast_fused_scale_softmax(self):
        for dtype in autocast_dtypes:
            msg = f"dtype: {dtype}"
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
            mask = None

            expected = torch_fn(attention_scores_1, mask)
            with torch.cuda.amp.autocast(dtype=dtype):
                actual = fused_fn(attention_scores_0, mask)
                self.assertEqual(actual.dtype, dtype, msg=msg)
            self.assertEqual(actual, expected, msg=msg)

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
            msg = f"{dtype}-{scale}-{softmax_in_fp32}"
            input_in_fp16 = dtype == torch.half
            input_in_bf16 = dtype == torch.bfloat16
            if not (scale is None or softmax_in_fp32):
                with self.assertRaises(RuntimeError, msg=msg):
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
            self.assertEqual(actual, expected, msg=msg)

            g0 = torch.randn_like(actual)
            with torch.no_grad():
                g1 = g0.clone()
            actual.backward(g0)
            expected.backward(g1)

    def test_autocast_fused_upper_triangle_mask_softmax(self):
        for dtype in autocast_dtypes:
            msg = f"dtype: {dtype}"
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
                self.assertEqual(actual.dtype, dtype, msg=msg)
            expected = torch_fn(attn_weights_1, total_mask)
            self.assertEqual(actual, expected, msg=msg)

            g0 = torch.randn_like(actual)
            with torch.no_grad():
                g1 = g0.clone()
            actual.backward(g0)
            expected.backward(g1)


class TestGenericFusedSoftmaxKernel(common_utils.TestCase):

    def setUp(self):
        super().setUp()
        self.batch = 2
        self.attn = 16
        self.scale_t = 1.0
        self.dtype = torch.float16
        self.device = torch.cuda.current_device()
        self.thresh = {"atol": 1e-3, "rtol": 1e-3}

        qlen = [1, 2]
        klen = [1, 2, 3, 4, 5, 8, 10, 11, 13, 128, 256, 1200, 1234]
        available_cuda_mem = torch.cuda.memory.mem_get_info(self.device)[0] / (1024 ** 3)
        if available_cuda_mem > 40:
            qlen.extend([1234, 2322, 2348])
            klen.extend([2048, 3123, 4096, 4128, 7234, 8192])

        self.q_k_lens = itertools.product(qlen, klen)

    def tearDown(self) -> None:
        torch.cuda.empty_cache()
        super().tearDown()

    def test_forward(self, allmasked: bool=False):
        import generic_scaled_masked_softmax_cuda
        for qlen, klen in self.q_k_lens:
            inputs = torch.normal(0, 2, (self.batch, self.attn, qlen, klen), dtype=self.dtype, device=self.device)
            masks = (
                torch.randint(0, 2, (self.batch, 1, qlen, klen), dtype=torch.bool, device=self.device)
                if not allmasked else torch.ones((self.batch, 1, qlen, klen), dtype=torch.bool, device=self.device)
            )
            softmax_results = generic_scaled_masked_softmax_cuda.forward(inputs, masks, self.scale_t)
            softmax_results_torch = forward_torch_softmax(inputs, masks, self.scale_t)
            self.assertEqual(
                softmax_results_torch.to(self.dtype), softmax_results, **self.thresh, msg=f"(q, k) = ({qlen, klen})")

    def test_backward(self, allmasked: bool=False):
        import generic_scaled_masked_softmax_cuda
        prev_thresh = self.thresh
        self.thresh = {"atol": 1.5e-1, "rtol": 5e-3}
        for qlen, klen in self.q_k_lens:
            inputs = torch.normal(0, 2, (self.batch, self.attn, qlen, klen), dtype=self.dtype, device=self.device)
            backward = torch.rand_like(inputs, dtype=torch.float16, device=self.device)
            masks = (
                torch.randint(0, 2, (self.batch, 1, qlen, klen), dtype=torch.bool, device=self.device)
                if not allmasked else torch.ones((self.batch, 1, qlen, klen), dtype=torch.bool, device=self.device)
            )
            softmax_results = generic_scaled_masked_softmax_cuda.forward(inputs, masks, self.scale_t)
            back_grad = generic_scaled_masked_softmax_cuda.backward(backward, softmax_results, self.scale_t)
            inputs.requires_grad = True
            softmax_results_torch = forward_torch_softmax(inputs, masks, self.scale_t)
            softmax_results_torch.backward(backward)
            self.assertEqual(back_grad, inputs.grad, **self.thresh, msg=f"(q, k) = ({qlen, klen})")
        self.thresh = prev_thresh

    def test_allmasked(self):
        self.test_forward(True)

    def test_allmask_backward(self):
        self.test_backward(True)


if __name__ == "__main__":
    common_utils.run_tests()
