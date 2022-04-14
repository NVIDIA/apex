import functools as func
import torch
import torch.distributed as dist
from torch import nn
import fast_bottleneck
import nccl_p2p_cuda as inc

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
        if spatial_group_size > 1:
            stream1 = spatial_halo_exchanger.stream1
            stream2 = spatial_halo_exchanger.stream2
            stream3 = spatial_halo_exchanger.stream3

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
        fast_bottleneck.forward_out1(explicit_nhwc, stride_1x1, args, outputs)

        if spatial_group_size > 1:
            out1 = outputs[0]
            if explicit_nhwc:
                N,Hs,W,C = list(out1.shape)
                memory_format = torch.contiguous_format
                out1_pad = torch.empty([N,Hs+2,W,C], dtype=out1.dtype, device='cuda')
            else:
                N,C,Hs,W = list(out1.shape)
                memory_format = torch.channels_last if out1.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
                out1_pad = torch.empty([N,C,Hs+2,W], dtype=out1.dtype, device='cuda', memory_format=memory_format)
            stream1.wait_stream(torch.cuda.current_stream())
            if spatial_method != 2: stream3.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream1):
                if explicit_nhwc:
                    top_out1_halo = out1_pad[:,:1,:,:]
                    btm_out1_halo = out1_pad[:,Hs+1:Hs+2,:,:]
                    spatial_halo_exchanger.left_right_halo_exchange(out1[:,:1,:,:], out1[:,Hs-1:,:,:], top_out1_halo, btm_out1_halo)
                else:
                    top_out1_halo = out1_pad[:,:,:1,:]
                    btm_out1_halo = out1_pad[:,:,Hs+1:Hs+2,:]
                    spatial_halo_exchanger.left_right_halo_exchange(out1[:,:,:1,:], out1[:,:,Hs-1:,:], top_out1_halo, btm_out1_halo)
            if spatial_method == 1:
                # overlap mid convolution with halo transfer
                if spatial_group_rank < spatial_group_size-1:
                    stream2.wait_stream(stream1)
                    with torch.cuda.stream(stream2):
                        if explicit_nhwc:
                            btm_fat_halo = torch.empty((N,3,W,C),dtype=out1.dtype,device=out1.device)
                            btm_fat_halo[:,0:2,:,:].copy_(out1[:,Hs-2:,:,:])
                            btm_fat_halo[:,2:,:,:].copy_(btm_out1_halo)
                        else:
                            btm_fat_halo = torch.empty((N,C,3,W),dtype=out1.dtype,device=out1.device)
                            btm_fat_halo[:,:,0:2,:].copy_(out1[:,:,Hs-2:,:])
                            btm_fat_halo[:,:,2:,:].copy_(btm_out1_halo)
                        btm_out2 = fast_bottleneck.forward_out2_halo(explicit_nhwc, btm_fat_halo, args)
                if spatial_group_rank > 0:
                    with torch.cuda.stream(stream1):
                        if explicit_nhwc:
                            top_fat_halo = torch.empty((N,3,W,C),dtype=out1.dtype,device=out1.device)
                            top_fat_halo[:,:1,:,:].copy_(top_out1_halo)
                            top_fat_halo[:,1:3,:,:].copy_(out1[:,:2,:,:])
                        else:
                            top_fat_halo = torch.empty((N,C,3,W),dtype=out1.dtype,device=out1.device)
                            top_fat_halo[:,:,:1,:].copy_(top_out1_halo)
                            top_fat_halo[:,:,1:3,:].copy_(out1[:,:,:2,:])
                        top_out2 = fast_bottleneck.forward_out2_halo(explicit_nhwc, top_fat_halo, args)
                if use_delay_kernel: inc.add_delay(10)
            elif spatial_method != 2 and spatial_method != 3:
                assert(False), "spatial_method must be 1, 2 or 3"

        if spatial_group_size <= 1:
            fast_bottleneck.forward_out2(explicit_nhwc, stride_1x1, args, outputs)
        elif spatial_method == 1:
            fast_bottleneck.forward_out2(explicit_nhwc, stride_1x1, args, outputs)
            with torch.cuda.stream(stream3):
                if explicit_nhwc:
                    out1_pad[:,1:Hs+1,:,:].copy_(out1)
                else:
                    out1_pad[:,:,1:Hs+1,:].copy_(out1)
        elif spatial_method == 2:
            # wait for halo transfer to finish before doing a full convolution of padded x
            if explicit_nhwc:
                out1_pad[:,1:Hs+1,:,:].copy_(out1)
            else:
                out1_pad[:,:,1:Hs+1,:].copy_(out1)
            torch.cuda.current_stream().wait_stream(stream1)
            fast_bottleneck.forward_out2_pad(explicit_nhwc, stride_1x1, args, outputs, out1_pad)
        elif spatial_method == 3:
            fast_bottleneck.forward_out2_mask(explicit_nhwc, stride_1x1, args, outputs, thresholdTop, thresholdBottom)
            with torch.cuda.stream(stream3):
                if explicit_nhwc:
                    out1_pad[:,1:Hs+1,:,:].copy_(out1)
                else:
                    out1_pad[:,:,1:Hs+1,:].copy_(out1)

        # compute halo cells for outputs[1] (out2)
        if spatial_group_size > 1:
            out2 = outputs[1]
            if explicit_nhwc:
                top_out2_halo = out2[:,:1,:,:]
                btm_out2_halo = out2[:,Hs-1:,:,:]
            else:
                top_out2_halo = out2[:,:,:1,:]
                btm_out2_halo = out2[:,:,Hs-1:,:]
            if spatial_method == 1:
                if spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(stream1)
                    top_out2_halo.copy_(top_out2)
                if spatial_group_rank < spatial_group_size-1:
                    torch.cuda.current_stream().wait_stream(stream2)
                    btm_out2_halo.copy_(btm_out2)
            elif spatial_method == 3:
                # Note
                # out2 halo correction cannot overlap with anything since it has
                # to wait for out2_mask to finish, but itself has to finish before
                # the first kernel of _forward_rest can launch.
                # At least we can overlap the two halo correction kernels.
                if spatial_group_rank < spatial_group_size-1:
                    stream2.wait_stream(stream1) # wait for halo transfers to finish
                    stream2.wait_stream(torch.cuda.current_stream()) # wait for *_out2_mask to finish
                    with torch.cuda.stream(stream2):
                        w1by3 = args[2][:,2:3,:,:].clone()
                        btm_out1_halo = btm_out1_halo.clone()
                        btm_out2 = fast_bottleneck.forward_out2_halo_corr(explicit_nhwc, btm_out1_halo, args, w1by3, btm_out2_halo.clone())
                        btm_out2_halo.copy_(btm_out2)
                if spatial_group_rank > 0:
                    stream1.wait_stream(torch.cuda.current_stream()) # wait for *_out2_mask to finish
                    with torch.cuda.stream(stream1):
                        w1by3 = args[2][:,:1,:,:].clone()
                        top_out1_halo = top_out1_halo.clone()
                        top_out2 = fast_bottleneck.forward_out2_halo_corr(explicit_nhwc, top_out1_halo, args, w1by3, top_out2_halo.clone())
                        top_out2_halo.copy_(top_out2)
                if spatial_group_rank < spatial_group_size-1:
                    torch.cuda.current_stream().wait_stream(stream2)
                if spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(stream1)
            
        fast_bottleneck.forward_rest(explicit_nhwc, stride_1x1, args, outputs)
        # save halos for backward pass
        if spatial_group_size > 1:
            if spatial_method != 2:
                # make sure copy of mid-section of out1 into out1_pad is done before exiting
                torch.cuda.current_stream().wait_stream(stream3)
            ctx.save_for_backward(*(args+outputs+[out1_pad,]))
        else:
            ctx.save_for_backward(*(args+outputs))
        # save relu outputs for drelu
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
            ctx.stream1 = stream1
            ctx.stream2 = stream2
            ctx.stream3 = stream3
        return outputs[2]

    # backward relu is not exposed, MUL with mask used now
    # only support dgrad
    @staticmethod
    def backward(ctx, grad_o):
        if ctx.spatial_group_size > 1:
            out1_pad = ctx.saved_tensors[-1]
            outputs = ctx.saved_tensors[-4:-1]
        else:
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

        grads = fast_bottleneck.backward_init(ctx.explicit_nhwc, ctx.stride_1x1, t_list)
        wgrad3_stream = torch.cuda.Stream()
        wgrad3_stream.wait_stream(torch.cuda.current_stream())
        grad_out2 = fast_bottleneck.backward_grad_out2(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads)
        wgrad2_stream = torch.cuda.Stream()
        wgrad2_stream.wait_stream(torch.cuda.current_stream())
        # do halo exchange of grad_out2 here
        # compute halo cells for grad_out1
        if ctx.spatial_group_size > 1:
            if ctx.explicit_nhwc:
                N,Hs,W,C = list(grad_out2.shape)
            else:
                N,C,Hs,W = list(grad_out2.shape)
            relu1 = t_list[12]
            ctx.stream1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ctx.stream1):
                top_halo, btm_halo = ctx.spatial_halo_exchanger.left_right_halo_exchange(grad_out2[:,:1,:,:], grad_out2[:,Hs-1:,:,:])
                # copy halos to send buffer
            if ctx.spatial_method == 1 or ctx.spatial_method == 2:
                # 1 -> halo recompute approach
                # 2 -> wait for concatenated halos, then do single conv on full input (not implemented yet for bprop)
                if ctx.spatial_group_rank < ctx.spatial_group_size-1:
                    ctx.stream2.wait_stream(ctx.stream1)
                    with torch.cuda.stream(ctx.stream2):
                        if ctx.explicit_nhwc:
                            btm_fat_halo = torch.empty((N,3,W,C),dtype=grad_out2.dtype,device=grad_out2.device)
                            btm_fat_halo[:,:2,:,:].copy_(grad_out2[:,Hs-2:,:,:])
                            btm_fat_halo[:,2:,:,:].copy_(btm_halo)
                            btm_fat_relu_halo = torch.empty((N,3,W,C),dtype=grad_out2.dtype,device=grad_out2.device)
                            btm_fat_relu_halo[:,:2,:,:].copy_(relu1[:,Hs-2:,:,:])
                            btm_fat_relu_halo[:,2:,:,:].zero_()
                        else:
                            btm_fat_halo = torch.empty((N,C,3,W),dtype=grad_out2.dtype,device=grad_out2.device)
                            btm_fat_halo[:,:,:2,:].copy_(grad_out2[:,:,Hs-2:,:])
                            btm_fat_halo[:,:,2:,:].copy_(btm_halo)
                            btm_fat_relu_halo = torch.empty((N,C,3,W),dtype=grad_out2.dtype,device=grad_out2.device)
                            btm_fat_relu_halo[:,:,:2,:].copy_(relu1[:,:,Hs-2:,:])
                            btm_fat_relu_halo[:,:,2:,:].zero_()
                        btm_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, btm_fat_halo, btm_fat_relu_halo)
                        if ctx.explicit_nhwc:
                            btm_grad_out1_halo = btm_grad_out1_halo[:,1:2,:,:]
                        else:
                            btm_grad_out1_halo = btm_grad_out1_halo[:,:,1:2,:]
                if ctx.spatial_group_rank > 0:
                    with torch.cuda.stream(ctx.stream1):
                        if ctx.explicit_nhwc:
                            top_fat_halo = torch.empty((N,3,W,C),dtype=grad_out2.dtype,device=grad_out2.device)
                            top_fat_halo[:,:1,:,:].copy_(top_halo)
                            top_fat_halo[:,1:,:,:].copy_(grad_out2[:,:2,:,:])
                            top_fat_relu_halo = torch.empty((N,3,W,C),dtype=grad_out2.dtype,device=grad_out2.device)
                            top_fat_relu_halo[:,:1,:,:].zero_()
                            top_fat_relu_halo[:,1:,:,:].copy_(relu1[:,:2,:,:])
                        else:
                            top_fat_halo = torch.empty((N,C,3,W),dtype=grad_out2.dtype,device=grad_out2.device)
                            top_fat_halo[:,:,:1,:].copy_(top_halo)
                            top_fat_halo[:,:,1:,:].copy_(grad_out2[:,:,:2,:])
                            top_fat_relu_halo = torch.empty((N,C,3,W),dtype=grad_out2.dtype,device=grad_out2.device)
                            top_fat_relu_halo[:,:,:1,:].zero_()
                            top_fat_relu_halo[:,:,1:,:].copy_(relu1[:,:,:2,:])
                        top_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, top_fat_halo, top_fat_relu_halo)
                        if ctx.explicit_nhwc:
                            top_grad_out1_halo = top_grad_out1_halo[:,1:2,:,:]
                        else:
                            top_grad_out1_halo = top_grad_out1_halo[:,:,1:2,:]
                if ctx.use_delay_kernel: inc.add_delay(10)
            elif ctx.spatial_method != 3:
                assert(False), "spatial_method must be 1, 2 or 3"

        # compute grad_out1 for internal cells
        if ctx.spatial_group_size <= 1 or ctx.spatial_method == 1 or ctx.spatial_method == 2:
            grad_out1 = fast_bottleneck.backward_grad_out1(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2)
        elif ctx.spatial_group_size > 1 and ctx.spatial_method == 3:
            grad_out1 = fast_bottleneck.backward_grad_out1_mask(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2, ctx.thresholdTop, ctx.thresholdBottom)

        # apply halo cells to grad_out1
        if ctx.spatial_group_size > 1:
            w = t_list[2]
            z = t_list[4]
            relu1 = t_list[12]
            #print("w.shape = %s, z.shape = %s, relu1.shape = %s" % (str(list(w.shape)), str(list(z.shape)), str(list(relu1.shape))))
            if ctx.spatial_method == 1 or ctx.spatial_method == 2:
                if ctx.spatial_group_rank < ctx.spatial_group_size-1:
                    torch.cuda.current_stream().wait_stream(ctx.stream2)
                    if ctx.explicit_nhwc:
                        grad_out1[:,Hs-1:,:,:].copy_(btm_grad_out1_halo)
                    else:
                        grad_out1[:,:,Hs-1:,:].copy_(btm_grad_out1_halo)
                    #print("ctx.spatial_group_rank = %d, apply grad_out1 btm halo (grad_out1.shape = %s)" % (ctx.spatial_group_rank, str(list(grad_out1.shape))))
                if ctx.spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(ctx.stream1)
                    if ctx.explicit_nhwc:
                        grad_out1[:,:1,:,:].copy_(top_grad_out1_halo)
                    else:
                        grad_out1[:,:,:1,:].copy_(top_grad_out1_halo)
                    #print("ctx.spatial_group_rank = %d, apply grad_out1 top halo (grad_out1.shape = %s)" % (ctx.spatial_group_rank, str(list(grad_out1.shape))))
            elif ctx.spatial_method == 3:
                if ctx.spatial_group_rank < ctx.spatial_group_size-1:
                    if ctx.explicit_nhwc:
                        btm_relu_halo = relu1[:,Hs-1:,:,:].clone()
                        btm_grad_out1 = grad_out1[:,Hs-1:,:,:]
                    else:
                        btm_relu_halo = relu1[:,:,Hs-1:,:].clone()
                        btm_grad_out1 = grad_out1[:,:,Hs-1:,:]
                    w1by3 = w[:,:1,:,:].clone()
                    ctx.stream2.wait_stream(ctx.stream1) # wait for halo transfers to finish
                    ctx.stream2.wait_stream(torch.cuda.current_stream()) # wait for backward_grad_out1_mask to finish before launching halo correction kernel
                    with torch.cuda.stream(ctx.stream2):
                        btm_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo_corr(ctx.explicit_nhwc, ctx.stride_1x1, t_list, w1by3, grads, btm_halo, btm_relu_halo, btm_grad_out1.clone())
                        btm_grad_out1.copy_(btm_grad_out1_halo)
                if ctx.spatial_group_rank > 0:
                    if ctx.explicit_nhwc:
                        top_relu_halo = relu1[:,:1,:,:].clone()
                        top_grad_out1 = grad_out1[:,:1,:,:]
                    else:
                        top_relu_halo = relu1[:,:,:1,:].clone()
                        top_grad_out1 = grad_out1[:,:,:1,:]
                    w1by3 = w[:,2:,:,:].clone()
                    ctx.stream1.wait_stream(torch.cuda.current_stream()) # wait for backward_grad_out1_mask to finish before launching halo correction kernel
                    with torch.cuda.stream(ctx.stream1):
                        top_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo_corr(ctx.explicit_nhwc, ctx.stride_1x1, t_list, w1by3, grads, top_halo, top_relu_halo, top_grad_out1.clone())
                        top_grad_out1.copy_(top_grad_out1_halo)
                if ctx.spatial_group_rank < ctx.spatial_group_size-1:
                    torch.cuda.current_stream().wait_stream(ctx.stream2) # wait for halo correction to finish
                if ctx.spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(ctx.stream1)

        wgrad1_stream = torch.cuda.Stream()
        wgrad1_stream.wait_stream(torch.cuda.current_stream())
        fast_bottleneck.backward_rest(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2, grad_out1)
        with torch.cuda.stream(wgrad3_stream):
            fast_bottleneck.backward_wgrad3(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads)
        with torch.cuda.stream(wgrad2_stream):
            if ctx.spatial_group_size > 1:
                fast_bottleneck.backward_wgrad2_pad(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, out1_pad, grad_out2)
            else:
                fast_bottleneck.backward_wgrad2(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2)
        with torch.cuda.stream(wgrad1_stream):
            fast_bottleneck.backward_wgrad1(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out1)
        torch.cuda.current_stream().wait_stream(wgrad3_stream)
        torch.cuda.current_stream().wait_stream(wgrad2_stream)
        torch.cuda.current_stream().wait_stream(wgrad1_stream)

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
                self.thresholdTop = torch.tensor([1 if spatial_group_rank > 0 else 0], dtype=torch.int32, device='cuda')
                self.thresholdBottom = torch.tensor([H-2 if spatial_group_rank < spatial_group_size - 1 else H-1], dtype=torch.int32, device='cuda')
            
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

