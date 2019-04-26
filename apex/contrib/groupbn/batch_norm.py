import torch
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm

import bnp

class bn_NHWC_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, b, rm, riv, mini_m, mini_riv, mom, epsilon, fuse_relu=False, is_train=True, bn_group=1, my_data=None, pair_data=None, magic=1, pair_data2=None):
        if is_train:
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.fuse_relu = fuse_relu
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.bn_group = bn_group

            res =  bnp.bn_fwd_nhwc(x, s, b, rm, riv, mini_m, mini_riv, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, bn_group, magic)
            return res
        else:
            return bnp.bn_fwd_eval_nhwc(x, s, b, rm, riv, bn_group, mom, epsilon, fuse_relu)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        fuse_relu = ctx.fuse_relu
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        bn_group = ctx.bn_group

        dx, dscale, dbias = bnp.bn_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, bn_group, magic)

        return dx, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None, None


class bn_addrelu_NHWC_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, s, b, rm, riv, mini_m, mini_riv, mom, epsilon, is_train=True, bn_group=1, my_data=None, pair_data=None, magic=1, pair_data2=None):
        if is_train:
            bitmask = torch.cuda.IntTensor(x.numel()//32)
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv, bitmask)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.bn_group = bn_group

            res =  bnp.bn_addrelu_fwd_nhwc(x, z, s, b, rm, riv, mini_m, mini_riv, bitmask, mom, epsilon, my_data, pair_data, pair_data2, bn_group, magic)
            return res
        else:
            return bnp.bn_addrelu_fwd_eval_nhwc(x, z, s, b, rm, riv, bn_group, mom, epsilon)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv, bitmask = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        bn_group = ctx.bn_group

        dx, dz, dscale, dbias = bnp.bn_addrelu_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, bitmask, mom, epsilon, my_data, pair_data, pair_data2, bn_group, magic)

        return dx, dz, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None





class BatchNorm2d_NHWC(_BatchNorm):
    def __init__(self, num_features, fuse_relu=False, bn_group=1):
        super(BatchNorm2d_NHWC, self).__init__(num_features)

        self.fuse_relu = fuse_relu

        self.minibatch_mean = torch.cuda.FloatTensor(num_features)
        self.minibatch_riv = torch.cuda.FloatTensor(num_features)

        #defaut to distributed bn disabled
        self.bn_group = bn_group
        self.my_data = None
        self.pair_data = None
        self.pair_data2 = None
        self.local_rank = 0
        self.magic = torch.IntTensor([0])

        #FIXME: turn pair handles into an array

        if bn_group>1:
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()          
            assert(world_size >= bn_group)
            assert(world_size % bn_group == 0)
         
            bn_sync_steps = 1
            if (bn_group==4):
                bn_sync_steps = 2

            self.ipc_buffer = torch.cuda.ByteTensor(bnp.get_buffer_size(bn_sync_steps))
            self.my_data = bnp.get_data_ptr(self.ipc_buffer)
            # we are walking on very thin ice here by utilizing internal `_share_cuda_()`
            self.storage = self.ipc_buffer.storage()
            self.share_cuda = self.storage._share_cuda_()
            internal_cuda_mem = self.share_cuda
            # internal_cuda_mem[1]: ipc_mem_handle
            my_handle = torch.cuda.ByteTensor(np.frombuffer(internal_cuda_mem[1], dtype=np.uint8))
            # internal_cuda_mem[3]: offset
            my_offset = torch.cuda.IntTensor([internal_cuda_mem[3]])

            handles_all = torch.empty(world_size, my_handle.size(0), dtype=my_handle.dtype, device=my_handle.device)
            handles_l = list(handles_all.unbind(0))
            torch.distributed.all_gather(handles_l, my_handle)

            offsets_all = torch.empty(world_size, my_offset.size(0), dtype=my_offset.dtype, device=my_offset.device)
            offsets_l = list(offsets_all.unbind(0))
            torch.distributed.all_gather(offsets_l, my_offset)

            #whom do I actually care about? that would be local_rank XOR 1
            self.pair_handle = handles_l[local_rank ^ 1].cpu().contiguous()
            pair_offset = offsets_l[local_rank ^ 1].cpu()
            self.pair_data = bnp.get_remote_data_ptr(self.pair_handle, pair_offset)

            if bn_group>2:
                self.pair_handle2 = handles_l[local_rank ^ 2].cpu().contiguous()
                pair_offset2 = offsets_l[local_rank ^ 2].cpu()
                self.pair_data2 = bnp.get_remote_data_ptr(self.pair_handle2, pair_offset2)

            #FIXME: get magic value into C code and eliminate from here
            self.magic = torch.IntTensor([2])
            self.local_rank = local_rank


    def forward(self, x, z=None):
        if z is not None:
            assert(self.fuse_relu==True)
            return bn_addrelu_NHWC_impl.apply(x, z,
                                  self.weight, self.bias,
                                  self.running_mean, self.running_var,
                                  self.minibatch_mean, self.minibatch_riv,
                                  self.momentum,
                                  self.eps, self.training, self.bn_group, self.my_data, self.pair_data, (self.magic), self.pair_data2)
        else:
            return bn_NHWC_impl.apply(x,
                                  self.weight, self.bias,
                                  self.running_mean, self.running_var,
                                  self.minibatch_mean, self.minibatch_riv,
                                  self.momentum,
                                  self.eps, self.fuse_relu, self.training, self.bn_group, self.my_data, self.pair_data, (self.magic), self.pair_data2)

    def __del__(self):
        if self.bn_group>1:
          bnp.close_remote_data(self.pair_handle)
          if self.bn_group>2:
              bnp.close_remote_data(self.pair_handle2)
