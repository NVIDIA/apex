import torch
import pdb
from torch.autograd import gradcheck
import fused_conv_bias_relu


class ConvBiasReLU_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, padding, stride):
        inputs = [x.half(), weight.half(), bias.half()]
        outputs = fused_conv_bias_relu.forward(inputs, padding, stride)
        ctx.save_for_backward(*(inputs[0:2] + outputs))
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
        inputs = [x.half(), weight.half(), bias.half(), mask.to(torch.int8)]
        outputs = fused_conv_bias_relu.forward_mask(inputs, padding, stride)
        ctx.save_for_backward(*(inputs[0:2] + outputs))
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
        inputs = [x.half(), weight.half(), bias.half()]
        outputs = fused_conv_bias_relu.forward_no_relu(inputs, padding, stride)
        ctx.save_for_backward(*(inputs[0:2]))
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


ConvBiasReLU = ConvBiasReLU_.apply
ConvBiasMaskReLU = ConvBiasMaskReLU_.apply
ConvBias = ConvBias_.apply

