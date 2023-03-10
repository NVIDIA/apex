import pdb

import torch
from torch.autograd import gradcheck

from apex import check_cudnn_version_and_warn
import fused_conv_bias_relu

check_cudnn_version_and_warn(__name__, 8400)


class ConvBiasReLU_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, padding, stride):
        outputs = fused_conv_bias_relu.forward([x, weight, bias], padding, stride)
        ctx.save_for_backward(x, weight, outputs[0])
        ctx.padding = padding
        ctx.stride = stride

        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward(bwd_args, padding, stride)

        return grads[0], grads[1], grads[2], None, None


class ConvBiasMaskReLU_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, mask, padding, stride):
        outputs = fused_conv_bias_relu.forward_mask([x, weight, bias, mask], padding, stride)
        ctx.save_for_backward(x, weight, outputs[0])
        ctx.padding = padding
        ctx.stride = stride

        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward(bwd_args, padding, stride)

        return grads[0], grads[1], grads[2], None, None, None


class ConvBias_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, padding, stride):
        outputs = fused_conv_bias_relu.forward_no_relu([x, weight, bias], padding, stride)
        ctx.save_for_backward(x, weight)
        ctx.padding = padding
        ctx.stride = stride

        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward_no_relu(bwd_args, padding, stride)

        return grads[0], grads[1], grads[2], None, None


class ConvFrozenScaleBiasReLU_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, scale, bias, padding, stride):
        output = fused_conv_bias_relu.forward_cscale_cbias_relu([x, weight, scale, bias], padding, stride)
        ctx.save_for_backward(x, weight, scale, output)
        ctx.padding = padding
        ctx.stride = stride

        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward_cscale_cbias_relu(bwd_args, padding, stride)

        return grads[0], grads[1], None, None, None, None


ConvBiasReLU = ConvBiasReLU_.apply
ConvBiasMaskReLU = ConvBiasMaskReLU_.apply
ConvBias = ConvBias_.apply
ConvFrozenScaleBiasReLU = ConvFrozenScaleBiasReLU_.apply

