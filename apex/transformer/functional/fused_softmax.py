# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda

        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
            inputs, scale_t[0]
        )

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None


def scaled_upper_triang_masked_softmax(inputs, _, scale):
    b, np, sq, sk = inputs.size()
    assert sq == sk, "causal mask is only for self attention"
    # Reshaping input to 3D tensor (attn_batches, sq, sk)
    inputs = inputs.view(-1, sq, sk)
    args = _cast_if_autocast_enabled(inputs, scale)
    with torch.cuda.amp.autocast(enabled=False):
        probs = ScaledUpperTriangMaskedSoftmax.apply(*args)
    return probs.view(b, np, sq, sk)


# NOTE (mkozuki): `ScaledMaskedSoftmax` somehow doesn't work well with `torch.cuda.amp.custom_fwd`.
# Without `cast_inputs` kwarg, somehow inputs are not cast to dtype used in the autocast context.
# So I needed to manually write two `torch.autograd.Function` inheritances.
# Fused operation which performs following three operations in sequence
# 1. Scale the tensor.
# 2. Apply the mask.
# 3. Perform softmax.
class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


def scaled_masked_softmax(inputs, mask, scale):
    # input is 4D tensor (b, np, sq, sk)
    if mask is not None:
        args = _cast_if_autocast_enabled(inputs, mask, scale)
        with torch.cuda.amp.autocast(enabled=False):
            return ScaledMaskedSoftmax.apply(*args)
    else:
        args = _cast_if_autocast_enabled(inputs, scale)
        with torch.cuda.amp.autocast(enabled=False):
            return ScaledSoftmax.apply(*args)


class GenericScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import generic_scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])
        softmax_results = generic_scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import generic_scaled_masked_softmax_cuda_new

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = generic_scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


def generic_scaled_masked_softmax(inputs, mask, scale):
    # input is 4D tensor (b, np, sq, sk)
    args = _cast_if_autocast_enabled(inputs, mask, scale)
    with torch.cuda.amp.autocast(enabled=False):
        return GenericScaledMaskedSoftmax.apply(*args)


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_softmax_cuda.forward(
            inputs, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


class FusedScaleMaskSoftmax(torch.nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        if self.input_in_fp16 and self.input_in_bf16:
            raise RuntimeError(
                "both fp16 and bf16 flags cannot be active at the same time."
            )
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        if not (self.scale is None or softmax_in_fp32):
            raise RuntimeError("softmax should be in fp32 when scaled")

        if self.scaled_masked_softmax_fusion:
            if self.attn_mask_type == AttnMaskType.causal:
                self.fused_softmax_func = scaled_upper_triang_masked_softmax
            elif self.attn_mask_type == AttnMaskType.padding:
                self.fused_softmax_func = scaled_masked_softmax
            else:
                raise ValueError("Invalid attn_mask_type.")

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4

        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and (
                self.attn_mask_type == AttnMaskType.causal
                or self.attn_mask_type == AttnMaskType.padding
            )
            and 16 < sk <= 16384  # sk must be 16 ~ 16384
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 16384:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, input, mask):
        # input.shape = [b, np, sq, sk]
        scale = self.scale if self.scale is not None else 1.0
        return self.fused_softmax_func(input, mask, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)

class GenericFusedScaleMaskSoftmax(FusedScaleMaskSoftmax):
    """
    Generic version of FusedSacleMaskSoftmax.
    It removes the seq-len limitations and has slight performance degragation compared with FusedScaleMaskSoftmax

    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self, input_in_fp16, input_in_bf16, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale,
    ):
        super().__init__(input_in_fp16, input_in_bf16, AttnMaskType.padding, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale)
        self.scaled_masked_softmax_fusion = generic_scaled_masked_softmax

    def is_kernel_available(self, mask, b, np, sq, sk):
        if self.scaled_masked_softmax_fusion and 0 < sk:  # user want to fuse  # sk must be 1 ~
            return True
        return False