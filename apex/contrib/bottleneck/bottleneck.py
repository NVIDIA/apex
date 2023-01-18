import functools as func

import torch
import torch.distributed as dist
from torch import nn

from apex import check_cudnn_version_and_warn
import fast_bottleneck
import nccl_p2p_cuda as inc


assert check_cudnn_version_and_warn(__name__, 8400)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    weight_tensor_nchw = tensor
    nn.init.kaiming_uniform_(weight_tensor_nchw, a=a, mode=mode, nonlinearity=nonlinearity)

def compute_scale_bias_one(nhwc, weight, bias, running_mean, running_var, w_scale, w_bias):
    scale = weight * running_var.rsqrt()
    bias = bias - running_mean * scale
    w_scale.copy_(scale)
    w_bias.copy_(bias)

def compute_scale_bias_method(nhwc, args):
    for arg in args:
        # arg is tuple of (weight, bias, running_mean, running_var, w_scale, w_bias)
        compute_scale_bias_one(nhwc, *arg)

class FrozenBatchNorm2d(torch.jit.ScriptModule):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    @torch.jit.script_method
    def get_scale_bias(self, nhwc):
        # type: (bool) -> List[torch.Tensor]
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        if nhwc:
            scale = scale.reshape(1, 1, 1, -1)
            bias = bias.reshape(1, 1, 1, -1)
        else:
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
        return scale, bias

    @torch.jit.script_method
    def forward(self, x):
        scale, bias = self.get_scale_bias(False)
        return x * scale + bias

@torch.jit.script
def drelu_dscale1(grad_o, output, scale1):
    relu_mask = (output>0)
    dx_relu = relu_mask * grad_o
    g1 = dx_relu * scale1
    return g1, dx_relu

@torch.jit.script
def drelu_dscale2(grad_o, output, scale1, scale2):
    relu_mask = (output>0)
    dx_relu = relu_mask * grad_o
    g1 = dx_relu * scale1
    g2 = dx_relu * scale2
    return g1, g2

class BottleneckFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nhwc, stride_1x1, scale, bias, x, *conv):
        # TODO: clean up order of tensors
        args = [x, *conv[0:3], *scale[0:3], *bias[0:3]]
        ctx.downsample = len(conv) > 3
        if ctx.downsample:
            args.append(conv[3])
            args.append(scale[3])
            args.append(bias[3])

        # weight buffers are always in nhwc while shape can be nhwc or channels_last
        # here we pass in flag and let c++ handle it
        # alternatively, we can put all sizes into a fixed format and pass it in
        outputs = fast_bottleneck.forward(nhwc, stride_1x1, args)
        ctx.save_for_backward(*(args+outputs))
        # save relu outputs for drelu
        ctx.nhwc = nhwc
        ctx.stride_1x1 = stride_1x1
        return outputs[2]

    # backward relu is not exposed, MUL with mask used now
    # only support dgrad
    @staticmethod
    def backward(ctx, grad_o):
        outputs = ctx.saved_tensors[-3:]

        if ctx.downsample:
            grad_conv3, grad_conv4 = drelu_dscale2(grad_o, outputs[2], ctx.saved_tensors[6], ctx.saved_tensors[11])
        else:
            grad_conv3, grad_conv4 = drelu_dscale1(grad_o, outputs[2], ctx.saved_tensors[6])

        # create input vector for backward
        t_list = [*ctx.saved_tensors[0:10]]
        t_list.append(grad_conv3)
        t_list.append(grad_conv4)

        # outputs used for wgrad and generating drelu mask
        t_list.append(outputs[0])
        t_list.append(outputs[1])

        # in case there is downsample
        if ctx.downsample:
            t_list.append(ctx.saved_tensors[10])

        grads = fast_bottleneck.backward(ctx.nhwc, ctx.stride_1x1, t_list)

        return (None, None, None, None, *grads)

bottleneck_function = BottleneckFunction.apply

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(torch.nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    # here we put it at 1x1

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, groups=1,
                 dilation=1, norm_func=None, use_cudnn=False, explicit_nhwc=False):
        super(Bottleneck, self).__init__()
        if groups != 1:
            raise RuntimeError('Only support groups == 1')
        if dilation != 1:
            raise RuntimeError('Only support dilation == 1')
        if norm_func == None:
            norm_func = FrozenBatchNorm2d
        else:
            raise RuntimeError('Only support frozen BN now.')

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_func(out_channels),
            )
        else:
            self.downsample = None

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, bottleneck_channels, stride)
        self.conv2 = conv3x3(bottleneck_channels, bottleneck_channels)
        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.bn1 = norm_func(bottleneck_channels)
        self.bn2 = norm_func(bottleneck_channels)
        self.bn3 = norm_func(out_channels)
        self.w_scale = None

        self.use_cudnn = use_cudnn

        # setup conv weights
        self.w_conv = [self.conv1.weight, self.conv2.weight, self.conv3.weight]
        if self.downsample is not None:
            self.w_conv.append(self.downsample[0].weight)

        # init weight in nchw format before possible transpose
        for w in self.w_conv:
            kaiming_uniform_(w, a=1)

        # TODO: prevent unsupported case usage
        # support cases
        #                 native      cudnn
        # normal             yes         no
        # channel_last       yes        yes
        # explicit_nhwc       no        yes
        self.explicit_nhwc = explicit_nhwc
        if self.explicit_nhwc:
            for p in self.parameters():
                with torch.no_grad():
                    p.data = p.data.permute(0,2,3,1).contiguous()

        return

    # Returns single callable that recomputes scale and bias for all frozen batch-norms.
    # This method must be called before cuda graphing.
    # The callable it returns can be called anytime.
    # Calling this method will prevent these from being computed every forward call.
    def get_scale_bias_callable(self):
        self.w_scale, self.w_bias, args = [], [], []
        batch_norms = [self.bn1, self.bn2, self.bn3]
        if self.downsample is not None:
            batch_norms.append(self.downsample[1])
        for bn in batch_norms:
            s = torch.empty_like(bn.weight)
            b = torch.empty_like(s)
            args.append( (bn.weight, bn.bias, bn.running_mean, bn.running_var, s, b) )
            if self.explicit_nhwc:
                self.w_scale.append( s.reshape(1, 1, 1, -1) )
                self.w_bias.append( b.reshape(1, 1, 1, -1) )
            else:
                self.w_scale.append( s.reshape(1, -1, 1, 1) )
                self.w_bias.append( b.reshape(1, -1, 1, 1) )
        return func.partial(compute_scale_bias_method, self.explicit_nhwc, args)

    def forward(self, x):
        if self.use_cudnn:
            if self.w_scale is None:
                # calculate scale/bias from registered buffers
                # TODO: make this better
                s1, b1 = self.bn1.get_scale_bias(self.explicit_nhwc)
                s2, b2 = self.bn2.get_scale_bias(self.explicit_nhwc)
                s3, b3 = self.bn3.get_scale_bias(self.explicit_nhwc)
                w_scale = [s1, s2, s3]
                w_bias = [b1, b2, b3]
                if self.downsample is not None:
                    s4, b4 = self.downsample[1].get_scale_bias(self.explicit_nhwc)
                    w_scale.append(s4)
                    w_bias.append(b4)
                out = bottleneck_function(self.explicit_nhwc, self.stride, w_scale, w_bias, x, *self.w_conv)
            else:
                out = bottleneck_function(self.explicit_nhwc, self.stride, self.w_scale, self.w_bias, x, *self.w_conv)
            return out

        if self.explicit_nhwc:
            raise RuntimeError('explicit nhwc with native ops is not supported.')

        # fallback to native ops
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SpatialBottleneckFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spatial_group_size, spatial_group_rank, spatial_communicator, spatial_halo_exchanger, spatial_method, use_delay_kernel, explicit_nhwc, stride_1x1, scale, bias, thresholdTop, thresholdBottom, x, *conv):
        assert spatial_group_size <= 1 or spatial_method in (1,2,3), "spatial_method must be 1, 2 or 3"

        main_stream = torch.cuda.current_stream()
        if spatial_group_size > 1:
            side_streams = [
                spatial_halo_exchanger.stream1,
                spatial_halo_exchanger.stream2,
            ]

        # TODO: clean up order of tensors
        args = [x, *conv[0:3], *scale[0:3], *bias[0:3]]
        ctx.downsample = len(conv) > 3
        if ctx.downsample:
            args.append(conv[3])
            args.append(scale[3])
            args.append(bias[3])

        # weight buffers are always in explicit_nhwc while shape can be explicit_nhwc or channels_last
        # here we pass in flag and let c++ handle it
        # alternatively, we can put all sizes into a fixed format and pass it in
        outputs = fast_bottleneck.forward_init(explicit_nhwc, stride_1x1, args)

        # 1x1 convolution (conv1)
        # Note: output into padded buffer if spatial parallelism is
        # enabled
        out1 = outputs[0]
        out1_pad = None
        if spatial_group_size > 1:
            if explicit_nhwc:
                N,Hs,W,C = list(out1.shape)
                out1_pad = torch.empty(
                    [N,Hs+2,W,C],
                    dtype=out1.dtype,
                    device='cuda',
                )
                out1 = out1_pad[:,1:-1,:,:]
            else:
                N,C,Hs,W = list(out1.shape)
                memory_format = (
                    torch.channels_last
                    if out1.is_contiguous(memory_format=torch.channels_last)
                    else torch.contiguous_format
                )
                out1_pad = torch.empty(
                    [N,C,Hs+2,W],
                    dtype=out1.dtype,
                    device='cuda',
                    memory_format=memory_format,
                )
                out1 = out1_pad[:,:,1:-1,:]
            outputs[0] = out1
        fast_bottleneck.conv_scale_bias_relu(
            [stride_1x1,stride_1x1],
            [0,0], # pre-pads
            [0,0], # post-pads
            [1,1], # dilations
            explicit_nhwc,
            args[0], # input
            args[1], # filter
            args[4], # scale
            args[7], # bias
            out1,
        )

        # 3x3 convolution (conv2) with spatial parallelism
        w2 = args[2]
        z2 = args[5]
        b2 = args[8]
        out2 = outputs[1]
        if spatial_group_size <= 1:
            fast_bottleneck.conv_scale_bias_relu(
                [1,1], # strides
                [1,1], # pre-pads
                [1,1], # post-pads
                [1,1], # dilations
                explicit_nhwc,
                out1,
                w2,
                z2,
                b2,
                out2,
            )
        else:

            # halo regions in input and output tensors
            if explicit_nhwc:
                top_out1_send = out1_pad[:,1:2,:,:]
                btm_out1_send = out1_pad[:,-2:-1,:,:]
                top_out1_recv = out1_pad[:,:1,:,:]
                btm_out1_recv = out1_pad[:,-1:,:,:]
                top_out2 = out2[:,:1,:,:]
                btm_out2 = out2[:,-1:,:,:]
            else:
                top_out1_send = out1_pad[:,:,1:2,:]
                btm_out1_send = out1_pad[:,:,-2:-1,:]
                top_out1_recv = out1_pad[:,:,:1,:]
                btm_out1_recv = out1_pad[:,:,-1:,:]
                top_out2 = out2[:,:,:1,:]
                btm_out2 = out2[:,:,-1:,:]

            # exchange halos
            side_streams[0].wait_stream(main_stream)
            side_streams[1].wait_stream(main_stream)
            with torch.cuda.stream(side_streams[0]):
                spatial_halo_exchanger.left_right_halo_exchange(
                    top_out1_send,
                    btm_out1_send,
                    top_out1_recv,
                    btm_out1_recv,
                )

            # overlap middle conv and halo convs
            if spatial_method == 1:

                # halo convolutions
                side_streams[1].wait_stream(side_streams[0])
                if spatial_group_rank > 0:
                    with torch.cuda.stream(side_streams[0]):
                        top_out1_fat = out1_pad[:,:3,:,:] if explicit_nhwc else out1_pad[:,:,:3,:]
                        fast_bottleneck.conv_scale_bias_relu(
                            [1,1], # strides
                            [0,1], # pre-pads
                            [0,1], # post-pads,
                            [1,1], # dilations
                            explicit_nhwc,
                            top_out1_fat,
                            w2,
                            z2,
                            b2,
                            top_out2,
                        )
                if spatial_group_rank < spatial_group_size-1:
                    with torch.cuda.stream(side_streams[1]):
                        btm_out1_fat = out1_pad[:,-3:,:,:] if explicit_nhwc else out1_pad[:,:,-3:,:]
                        fast_bottleneck.conv_scale_bias_relu(
                            [1,1], # strides
                            [0,1], # pre-pads
                            [0,1], # post-pads,
                            [1,1], # dilations
                            explicit_nhwc,
                            btm_out1_fat,
                            w2,
                            z2,
                            b2,
                            btm_out2,
                        )

                # manual delay to improve kernel overlapping
                if use_delay_kernel:
                    inc.add_delay(10)

                # middle convolution
                pre_pads = [1, 1]
                post_pads = [1, 1]
                middle_out2 = out2
                if spatial_group_rank > 0:
                    pre_pads[0] = 0
                    middle_out2 = middle_out2[:,1:,:,:] if explicit_nhwc else middle_out2[:,:,1:,:]
                if spatial_group_rank < spatial_group_size-1:
                    post_pads[0] = 0
                    middle_out2 = middle_out2[:,:-1,:,:] if explicit_nhwc else middle_out2[:,:,:-1,:]
                fast_bottleneck.conv_scale_bias_relu(
                    [1,1], # strides
                    pre_pads,
                    post_pads,
                    [1,1], # dilations
                    explicit_nhwc,
                    out1,
                    w2,
                    z2,
                    b2,
                    middle_out2,
                )

            # blocking halo exchange
            if spatial_method == 2:
                main_stream.wait_stream(side_streams[0])
                fast_bottleneck.conv_scale_bias_relu(
                    [1,1], # strides
                    [0,1], # pre-pads
                    [0,1], # post-pads
                    [1,1], # dilations
                    explicit_nhwc,
                    out1_pad,
                    w2,
                    z2,
                    b2,
                    out2,
                )

            # partial conv with halo correction
            if spatial_method == 3:

                # middle convolution
                fast_bottleneck.conv_scale_bias_relu_mask(
                    [1,1], # strides
                    [1,1], # pre-pads
                    [1,1], # post-pads
                    [1,1], # dilations
                    explicit_nhwc,
                    1, # mask axis
                    out1,
                    w2,
                    z2,
                    b2,
                    out2,
                    thresholdTop,
                    thresholdBottom,
                )
                if spatial_group_rank > 0:
                    top_out2_partial = top_out2.clone()
                if spatial_group_rank < spatial_group_size-1:
                    btm_out2_partial = btm_out2.clone()

                # make contiguous copy of conv kernel section
                with torch.cuda.stream(side_streams[1]):
                    if spatial_group_rank > 0:
                        top_w2 = w2[:,:1,:,:] if explicit_nhwc else w2[:,:,:1,:]
                        top_w2 = top_w2.clone()
                    if spatial_group_rank < spatial_group_size-1:
                        btm_w2 = w2[:,-1:,:,:] if explicit_nhwc else w2[:,:,-1:,:]
                        btm_w2 = btm_w2.clone()

                # halo correction
                side_streams[0].wait_stream(main_stream)
                side_streams[0].wait_stream(side_streams[1])
                side_streams[1].wait_stream(side_streams[0])
                if spatial_group_rank > 0:
                    with torch.cuda.stream(side_streams[0]):
                        fast_bottleneck.conv_add_scale_bias_relu(
                            [1,1], # strides
                            [0,1], # pre-pads
                            [0,1], # post-pads
                            [1,1], # dilations
                            explicit_nhwc,
                            top_out1_recv,
                            top_w2,
                            top_out2_partial,
                            z2,
                            b2,
                            top_out2,
                        )
                if spatial_group_rank < spatial_group_size-1:
                    with torch.cuda.stream(side_streams[1]):
                        fast_bottleneck.conv_add_scale_bias_relu(
                            [1,1], # strides
                            [0,1], # pre-pads
                            [0,1], # post-pads
                            [1,1], # dilations
                            explicit_nhwc,
                            btm_out1_recv,
                            btm_w2,
                            btm_out2_partial,
                            z2,
                            b2,
                            btm_out2,
                        )

            # synchronize streams
            main_stream.wait_stream(side_streams[0])
            main_stream.wait_stream(side_streams[1])

        # 1x1 convolution (conv3) and residual connection (maybe conv4)
        fast_bottleneck.forward_rest(explicit_nhwc, stride_1x1, args, outputs)

        # save state for backward pass
        ctx.save_for_backward(*(args+outputs+[out1_pad,]))
        ctx.explicit_nhwc = explicit_nhwc
        ctx.stride_1x1 = stride_1x1
        ctx.spatial_group_size = spatial_group_size
        if spatial_group_size > 1:
            ctx.spatial_group_rank = spatial_group_rank
            ctx.spatial_halo_exchanger = spatial_halo_exchanger
            ctx.spatial_method = spatial_method
            ctx.use_delay_kernel = use_delay_kernel
            ctx.thresholdTop = thresholdTop
            ctx.thresholdBottom = thresholdBottom
            ctx.side_streams = side_streams
        return outputs[2]

    # backward relu is not exposed, MUL with mask used now
    # only support dgrad
    @staticmethod
    def backward(ctx, grad_o):
        explicit_nhwc = ctx.explicit_nhwc
        stride_1x1 = ctx.stride_1x1
        spatial_group_size = ctx.spatial_group_size
        if spatial_group_size > 1:
            spatial_group_rank = ctx.spatial_group_rank
            spatial_halo_exchanger = ctx.spatial_halo_exchanger
            spatial_method = ctx.spatial_method
            use_delay_kernel = ctx.use_delay_kernel
            thresholdTop = ctx.thresholdTop
            thresholdBottom = ctx.thresholdBottom
            halo_streams = ctx.side_streams

        outputs = ctx.saved_tensors[-4:-1]
        out1_pad = ctx.saved_tensors[-1]

        main_stream = torch.cuda.current_stream()
        wgrad_streams = [
            torch.cuda.Stream(),
            torch.cuda.Stream(),
            torch.cuda.Stream(),
        ]

        if ctx.downsample:
            grad_conv3, grad_conv4 = drelu_dscale2(grad_o, outputs[2], ctx.saved_tensors[6], ctx.saved_tensors[11])
        else:
            grad_conv3, grad_conv4 = drelu_dscale1(grad_o, outputs[2], ctx.saved_tensors[6])

        # create input vector for backward
        t_list = [*ctx.saved_tensors[0:10]]
        t_list.append(grad_conv3)
        t_list.append(grad_conv4)

        # outputs used for wgrad and generating drelu mask
        t_list.append(outputs[0])
        t_list.append(outputs[1])

        # in case there is downsample
        if ctx.downsample:
            t_list.append(ctx.saved_tensors[10])

        # initialize gradient buffers
        grads = fast_bottleneck.backward_init(explicit_nhwc, stride_1x1, t_list)

        # 1x1 convolution (conv3)
        wgrad_streams[0].wait_stream(main_stream)
        out2 = outputs[1]
        if spatial_group_size <= 1:
            grad_out2 = torch.empty_like(out2)
            grad_out2_pad = None
        else:
            if explicit_nhwc:
                N,Hs,W,C = list(out2.shape)
                grad_out2_pad = torch.empty(
                    [N,Hs+2,W,C],
                    dtype=out2.dtype,
                    device='cuda',
                )
                grad_out2 = grad_out2_pad[:,1:-1,:,:]
            else:
                N,C,Hs,W = list(out2.shape)
                memory_format = (
                    torch.channels_last
                    if out2.is_contiguous(memory_format=torch.channels_last)
                    else torch.contiguous_format
                )
                grad_out2_pad = torch.empty(
                    [N,C,Hs+2,W],
                    dtype=out2.dtype,
                    device='cuda',
                    memory_format=memory_format,
                )
                grad_out2 = grad_out2_pad[:,:,1:-1,:]
        fast_bottleneck.backward_grad_out2(explicit_nhwc, stride_1x1, t_list, grads, grad_out2)

        # 3x3 convolution (conv2) with spatial parallelism
        wgrad_streams[1].wait_stream(main_stream)
        out1 = outputs[0]
        grad_out1 = torch.empty_like(out1)
        w2 = t_list[2];
        z1 = t_list[4];
        if spatial_group_size <= 1:
            fast_bottleneck.dconv_drelu_dscale(
                [1,1], # strides
                [1,1], # pre-pads
                [1,1], # post-pads
                [1,1], # dilations
                explicit_nhwc,
                grad_out2,
                out1,
                w2,
                z1,
                grad_out1,
            )
        else:
            if explicit_nhwc:
                N,Hs,W,C = list(grad_out2.shape)
                top_grad_out2_send = grad_out2_pad[:,1:2,:,:]
                btm_grad_out2_send = grad_out2_pad[:,-2:-1:,:,:]
                top_grad_out2_recv = grad_out2_pad[:,:1,:,:]
                btm_grad_out2_recv = grad_out2_pad[:,-1:,:,:]
                top_grad_out1 = grad_out1[:,:1,:,:]
                btm_grad_out1 = grad_out1[:,-1:,:,:]
                top_out1 = out1[:,:1,:,:]
                btm_out1 = out1[:,-1:,:,:]
            else:
                N,C,Hs,W = list(grad_out2.shape)
                top_grad_out2_send = grad_out2_pad[:,:,1:2,:]
                btm_grad_out2_send = grad_out2_pad[:,:,-2:-1:,:]
                top_grad_out2_recv = grad_out2_pad[:,:,:1,:]
                btm_grad_out2_recv = grad_out2_pad[:,:,-1:,:]
                top_grad_out1 = grad_out1[:,:,:1,:]
                btm_grad_out1 = grad_out1[:,:,-1:,:]
                top_out1 = out1[:,:,:1,:]
                btm_out1 = out1[:,:,-1:,:]

            # exchange halos
            halo_streams[0].wait_stream(main_stream)
            halo_streams[1].wait_stream(main_stream)
            with torch.cuda.stream(halo_streams[0]):
                spatial_halo_exchanger.left_right_halo_exchange(
                    top_grad_out2_send,
                    btm_grad_out2_send,
                    top_grad_out2_recv,
                    btm_grad_out2_recv,
                )

            # overlap middle conv and halo convs
            if spatial_method == 1:

                # halo convolutions
                halo_streams[1].wait_stream(halo_streams[0])
                if spatial_group_rank > 0:
                    with torch.cuda.stream(halo_streams[0]):
                        if explicit_nhwc:
                            top_grad_out2_fat = grad_out2_pad[:,:3,:,:]
                        else:
                            top_grad_out2_fat = grad_out2_pad[:,:,:3,:]
                        fast_bottleneck.dconv_drelu_dscale(
                            [1,1], # strides
                            [2,1], # pre-pads
                            [2,1], # post-pads,
                            [1,1], # dilations
                            explicit_nhwc,
                            top_grad_out2_fat,
                            top_out1,
                            w2,
                            z1,
                            top_grad_out1,
                        )
                if spatial_group_rank < spatial_group_size-1:
                    with torch.cuda.stream(halo_streams[1]):
                        if explicit_nhwc:
                            btm_grad_out2_fat = grad_out2_pad[:,-3:,:,:]
                        else:
                            btm_grad_out2_fat = grad_out2_pad[:,:,-3:,:]
                        fast_bottleneck.dconv_drelu_dscale(
                            [1,1], # strides
                            [2,1], # pre-pads
                            [2,1], # post-pads,
                            [1,1], # dilations
                            explicit_nhwc,
                            btm_grad_out2_fat,
                            btm_out1,
                            w2,
                            z1,
                            btm_grad_out1,
                        )

                # manual delay to improve kernel overlapping
                if use_delay_kernel: inc.add_delay(10)

                # middle convolution
                pre_pads = [1, 1]
                post_pads = [1, 1]
                middle_out1 = out1
                middle_grad_out1 = grad_out1
                if spatial_group_rank > 0:
                    pre_pads[0] = 2
                    if explicit_nhwc:
                        middle_out1 = middle_out1[:,1:,:,:]
                        middle_grad_out1 = middle_grad_out1[:,1:,:,:]
                    else:
                        middle_out1 = middle_out1[:,:,1:,:]
                        middle_grad_out1 = middle_grad_out1[:,:,1:,:]
                if spatial_group_rank < spatial_group_size-1:
                    post_pads[0] = 2
                    if explicit_nhwc:
                        middle_out1 = middle_out1[:,:-1,:,:]
                        middle_grad_out1 = middle_grad_out1[:,:-1,:,:]
                    else:
                        middle_out1 = middle_out1[:,:,:-1,:]
                        middle_grad_out1 = middle_grad_out1[:,:,:-1,:]
                fast_bottleneck.dconv_drelu_dscale(
                    [1,1], # strides
                    pre_pads,
                    post_pads,
                    [1,1], # dilations
                    explicit_nhwc,
                    grad_out2,
                    middle_out1,
                    w2,
                    z1,
                    middle_grad_out1,
                )

            # blocking halo exchange
            if spatial_method == 2:
                main_stream.wait_stream(halo_streams[0])
                fast_bottleneck.dconv_drelu_dscale(
                    [1,1], # strides
                    [2,1], # pre-pads
                    [2,1], # post-pads,
                    [1,1], # dilations
                    explicit_nhwc,
                    grad_out2_pad,
                    out1,
                    w2,
                    z1,
                    grad_out1,
                )

            # partial conv with halo correction
            if spatial_method == 3:

                # middle convolution
                fast_bottleneck.dconv_drelu_dscale_mask(
                    [1,1], # strides
                    [1,1], # pre-pads
                    [1,1], # post-pads,
                    [1,1], # dilations
                    explicit_nhwc,
                    1, # mask axis
                    grad_out2,
                    out1,
                    w2,
                    z1,
                    grad_out1,
                    thresholdTop,
                    thresholdBottom,
                )
                if spatial_group_rank > 0:
                    top_grad_out1_partial = top_grad_out1.clone()
                if spatial_group_rank < spatial_group_size-1:
                    btm_grad_out1_partial = btm_grad_out1.clone()

                # make contiguous copy of conv kernel section
                with torch.cuda.stream(halo_streams[1]):
                    if spatial_group_rank > 0:
                        top_w2 = w2[:,:1,:,:] if explicit_nhwc else w2[:,:,:1,:]
                        top_w2 = top_w2.clone()
                    if spatial_group_rank < spatial_group_size-1:
                        btm_w2 = w2[:,-1:,:,:] if explicit_nhwc else w2[:,:,-1:,:]
                        btm_w2 = btm_w2.clone()

                # halo correction
                halo_streams[0].wait_stream(main_stream)
                halo_streams[0].wait_stream(halo_streams[1])
                halo_streams[1].wait_stream(halo_streams[0])
                if spatial_group_rank > 0:
                    with torch.cuda.stream(halo_streams[0]):
                        fast_bottleneck.dconv_add_drelu_dscale(
                            [1,1], # strides
                            [0,1], # pre-pads
                            [0,1], # post-pads
                            [1,1], # dilations
                            explicit_nhwc,
                            top_grad_out2_recv,
                            top_out1,
                            top_w2,
                            top_grad_out1_partial,
                            z1,
                            top_grad_out1,
                        )
                if spatial_group_rank < spatial_group_size-1:
                    with torch.cuda.stream(halo_streams[1]):
                        fast_bottleneck.dconv_add_drelu_dscale(
                            [1,1], # strides
                            [0,1], # pre-pads
                            [0,1], # post-pads
                            [1,1], # dilations
                            explicit_nhwc,
                            btm_grad_out2_recv,
                            btm_out1,
                            btm_w2,
                            btm_grad_out1_partial,
                            z1,
                            btm_grad_out1,
                        )

            # synchronize streams
            main_stream.wait_stream(halo_streams[0])
            main_stream.wait_stream(halo_streams[1])

        # 1x1 convolution (conv1)
        wgrad_streams[2].wait_stream(main_stream)
        fast_bottleneck.backward_rest(explicit_nhwc, stride_1x1, t_list, grads, grad_out2, grad_out1)

        # weight gradients
        with torch.cuda.stream(wgrad_streams[0]):
            fast_bottleneck.backward_wgrad3(explicit_nhwc, stride_1x1, t_list, grads)
        with torch.cuda.stream(wgrad_streams[1]):
            if spatial_group_size > 1:
                fast_bottleneck.backward_wgrad2_pad(explicit_nhwc, stride_1x1, t_list, grads, out1_pad, grad_out2)
            else:
                fast_bottleneck.backward_wgrad2(explicit_nhwc, stride_1x1, t_list, grads, grad_out2)
        with torch.cuda.stream(wgrad_streams[2]):
            fast_bottleneck.backward_wgrad1(explicit_nhwc, stride_1x1, t_list, grads, grad_out1)
        for stream in wgrad_streams:
            main_stream.wait_stream(stream)

        return (None, None, None, None, None, None, None, None, None, None, None, None, *grads)

spatial_bottleneck_function = SpatialBottleneckFunction.apply

class SpatialBottleneck(torch.nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    # here we put it at 1x1

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, groups=1,
                 dilation=1, norm_func=None, use_cudnn=False, explicit_nhwc=False,
                 spatial_parallel_args=None):
        super(SpatialBottleneck, self).__init__()
        if groups != 1:
            raise RuntimeError('Only support groups == 1')
        if dilation != 1:
            raise RuntimeError('Only support dilation == 1')
        if norm_func == None:
            norm_func = FrozenBatchNorm2d
        else:
            raise RuntimeError('Only support frozen BN now.')

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_func(out_channels),
            )
        else:
            self.downsample = None

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, bottleneck_channels, stride)
        self.conv2 = conv3x3(bottleneck_channels, bottleneck_channels)
        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.bn1 = norm_func(bottleneck_channels)
        self.bn2 = norm_func(bottleneck_channels)
        self.bn3 = norm_func(out_channels)
        self.w_scale = None

        self.use_cudnn = use_cudnn

        # setup conv weights
        self.w_conv = [self.conv1.weight, self.conv2.weight, self.conv3.weight]
        if self.downsample is not None:
            self.w_conv.append(self.downsample[0].weight)

        # init weight in nchw format before possible transpose
        for w in self.w_conv:
            kaiming_uniform_(w, a=1)

        self.thresholdTop, self.thresholdBottom = None, None

        # TODO: prevent unsupported case usage
        # support cases
        #                 native      cudnn
        # normal             yes         no
        # channel_last       yes        yes
        # explicit_nhwc       no        yes
        self.explicit_nhwc = explicit_nhwc
        if self.explicit_nhwc:
            for p in self.parameters():
                with torch.no_grad():
                    p.data = p.data.permute(0,2,3,1).contiguous()

        # spatial communicator
        if spatial_parallel_args is None:
            self.spatial_parallel_args = (1, 0, None, None, 0, False)
        else:
            self.spatial_parallel_args = spatial_parallel_args
        return

    # Returns single callable that recomputes scale and bias for all frozen batch-norms.
    # This method must be called before cuda graphing.
    # The callable it returns can be called anytime.
    # Calling this method will prevent these from being computed every forward call.
    def get_scale_bias_callable(self):
        self.w_scale, self.w_bias, args = [], [], []
        batch_norms = [self.bn1, self.bn2, self.bn3]
        if self.downsample is not None:
            batch_norms.append(self.downsample[1])
        for bn in batch_norms:
            s = torch.empty_like(bn.weight)
            b = torch.empty_like(s)
            args.append( (bn.weight, bn.bias, bn.running_mean, bn.running_var, s, b) )
            if self.explicit_nhwc:
                self.w_scale.append( s.reshape(1, 1, 1, -1) )
                self.w_bias.append( b.reshape(1, 1, 1, -1) )
            else:
                self.w_scale.append( s.reshape(1, -1, 1, 1) )
                self.w_bias.append( b.reshape(1, -1, 1, 1) )
        return func.partial(compute_scale_bias_method, self.explicit_nhwc, args)

    def forward(self, x):
        if self.use_cudnn:
            if self.thresholdTop is None:
                spatial_group_size, spatial_group_rank, _, _, _, _ = self.spatial_parallel_args
                if self.explicit_nhwc:
                    N,H,W,C = list(x.shape)
                else:
                    N,C,H,W = list(x.shape)
                self.thresholdTop = torch.tensor([0 if spatial_group_rank > 0 else -1], dtype=torch.int32, device='cuda')
                self.thresholdBottom = torch.tensor([H-1 if spatial_group_rank < spatial_group_size - 1 else H], dtype=torch.int32, device='cuda')

            if self.w_scale is None:
                # calculate scale/bias from registered buffers
                # TODO: make this better
                s1, b1 = self.bn1.get_scale_bias(self.explicit_nhwc)
                s2, b2 = self.bn2.get_scale_bias(self.explicit_nhwc)
                s3, b3 = self.bn3.get_scale_bias(self.explicit_nhwc)
                w_scale = [s1, s2, s3]
                w_bias = [b1, b2, b3]
                if self.downsample is not None:
                    s4, b4 = self.downsample[1].get_scale_bias(self.explicit_nhwc)
                    w_scale.append(s4)
                    w_bias.append(b4)
                out = spatial_bottleneck_function(*self.spatial_parallel_args, self.explicit_nhwc, self.stride, w_scale, w_bias, self.thresholdTop, self.thresholdBottom, x, *self.w_conv)
            else:
                out = spatial_bottleneck_function(*self.spatial_parallel_args, self.explicit_nhwc, self.stride, self.w_scale, self.w_bias, self.thresholdTop, self.thresholdBottom, x, *self.w_conv)
            return out

        if self.explicit_nhwc:
            raise RuntimeError('explicit nhwc with native ops is not supported.')

        # fallback to native ops
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
