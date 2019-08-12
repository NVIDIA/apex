import torch
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm

import bnp

class bn_NHWC_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, is_train, bn_group, my_data, pair_data, magic, pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup, bwd_grid_x, multi_stream):
        if is_train:
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.ret_cta = ret_cta
            ctx.fuse_relu = fuse_relu
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.pair_data3 = pair_data3
            ctx.bn_group = bn_group
            ctx.bwd_occup = bwd_occup
            ctx.bwd_grid_x = bwd_grid_x
            ctx.multi_stream = multi_stream

            res =  bnp.bn_fwd_nhwc(x, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, fwd_occup, fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_fwd_eval_nhwc(x, s, b, rm, riv, ret_cta, bn_group, mom, epsilon, fuse_relu)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        ret_cta = ctx.ret_cta
        fuse_relu = ctx.fuse_relu
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        pair_data3 = ctx.pair_data3
        bn_group = ctx.bn_group
        bwd_occup = ctx.bwd_occup
        bwd_grid_x = ctx.bwd_grid_x
        multi_stream = ctx.multi_stream

        dx, dscale, dbias = bnp.bn_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup, bwd_grid_x, multi_stream)

        return dx, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class bn_addrelu_NHWC_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, s, b, rm, riv, mini_m, mini_riv, grid_dim_y, ret_cta, mom, epsilon, is_train, bn_group, my_data, pair_data, magic, pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup, bwd_grid_x, multi_stream):
        if is_train:
            bitmask = torch.cuda.IntTensor(((x.numel()+31)//32) * 2 * grid_dim_y)
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv, bitmask)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.ret_cta = ret_cta
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.pair_data3 = pair_data3
            ctx.bn_group = bn_group
            ctx.bwd_occup = bwd_occup
            ctx.bwd_grid_x = bwd_grid_x
            ctx.multi_stream = multi_stream

            res =  bnp.bn_addrelu_fwd_nhwc(x, z, s, b, rm, riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, fwd_occup, fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_addrelu_fwd_eval_nhwc(x, z, s, b, rm, riv, ret_cta, bn_group, mom, epsilon)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv, bitmask = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        ret_cta = ctx.ret_cta
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        pair_data3 = ctx.pair_data3
        bn_group = ctx.bn_group
        bwd_occup = ctx.bwd_occup
        bwd_grid_x = ctx.bwd_grid_x
        multi_stream = ctx.multi_stream

        dx, dz, dscale, dbias = bnp.bn_addrelu_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup, bwd_grid_x, multi_stream)

        return dx, dz, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None





class BatchNorm2d_NHWC(_BatchNorm):
    # if using BatchNorm2d_NHWC simultaneously with multiple streams set multi_stream to True
    def __init__(self, num_features, fuse_relu=False, bn_group=1, max_cta_per_sm=2, cta_launch_margin=12, multi_stream=False):
        super(BatchNorm2d_NHWC, self).__init__(num_features)

        self.fuse_relu = fuse_relu
        self.multi_stream = multi_stream

        self.minibatch_mean = torch.cuda.FloatTensor(num_features)
        self.minibatch_riv = torch.cuda.FloatTensor(num_features)

        #defaut to distributed bn disabled
        self.bn_group = bn_group
        self.max_cta_per_sm = max_cta_per_sm        #used only in training fwd and bwd
        self.cta_launch_margin = cta_launch_margin  #used only in training fwd and bwd
        self.my_data = None
        self.pair_data = None
        self.pair_data2 = None
        self.pair_data3 = None
        self.local_rank = 0
        self.magic = torch.IntTensor([0])

        #calculate cta per sm occupancies
        assert(max_cta_per_sm>0) # won't be able to do much with 0 CTAs :)
        self.fwd_occupancy =  min(bnp.bn_fwd_nhwc_occupancy(), max_cta_per_sm)
        self.bwd_occupancy =  min(bnp.bn_bwd_nhwc_occupancy(), max_cta_per_sm)
        self.addrelu_fwd_occupancy =  min(bnp.bn_addrelu_fwd_nhwc_occupancy(), max_cta_per_sm)
        self.addrelu_bwd_occupancy =  min(bnp.bn_addrelu_bwd_nhwc_occupancy(), max_cta_per_sm)

        #calculate grid dimentions based on occupancy numbers
        mp_count = torch.cuda.get_device_properties(None).multi_processor_count
        self.fwd_grid_dim_x = max(mp_count*self.fwd_occupancy - cta_launch_margin , 1)
        self.bwd_grid_dim_x = max(mp_count*self.bwd_occupancy - cta_launch_margin , 1)
        self.addrelu_fwd_grid_dim_x = max(mp_count*self.addrelu_fwd_occupancy - cta_launch_margin , 1)
        self.addrelu_bwd_grid_dim_x = max(mp_count*self.addrelu_bwd_occupancy - cta_launch_margin , 1)
        self.grid_dim_y = (num_features + 63) // 64

        # allocate scratch space used by implementation
        # TODO: scratch space that is not supposed to be exposed at user code. We only need one time initialization, the
        # same buffer could be reused in future iterations. Currently we exposed it here instead of requesting new
        # buffer from cache allocator to avoid unnecessary initialization at future iterations.
        self.ret_cta = torch.cuda.ByteTensor(8192).fill_(0)

        #FIXME: turn pair handles into an array
        if bn_group>1:
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()          
            assert(world_size >= bn_group)
            assert(world_size % bn_group == 0)
         
            bn_sync_steps = 1
            if (bn_group==4):
                bn_sync_steps = 2
            if (bn_group==8):
                bn_sync_steps = 3

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

            if bn_group>4:
                self.pair_handle3 = handles_l[local_rank ^ 4].cpu().contiguous()
                pair_offset3 = offsets_l[local_rank ^ 4].cpu()
                self.pair_data3 = bnp.get_remote_data_ptr(self.pair_handle3, pair_offset3)

            #FIXME: get magic value into C code and eliminate from here
            self.magic = torch.IntTensor([2])
            self.local_rank = local_rank


    def forward(self, x, z=None):
        if z is not None:
            assert(self.fuse_relu==True)
            return bn_addrelu_NHWC_impl.apply(x, z,
                                  self.weight, self.bias,
                                  self.running_mean, self.running_var,
                                  self.minibatch_mean, self.minibatch_riv, self.grid_dim_y, self.ret_cta,
                                  self.momentum,
                                  self.eps, self.training, self.bn_group, self.my_data, self.pair_data, (self.magic), self.pair_data2, self.pair_data3,
                                  self.addrelu_fwd_occupancy, self.addrelu_fwd_grid_dim_x,
                                  self.addrelu_bwd_occupancy, self.addrelu_bwd_grid_dim_x,
                                  self.multi_stream)
        else:
            return bn_NHWC_impl.apply(x,
                                  self.weight, self.bias,
                                  self.running_mean, self.running_var,
                                  self.minibatch_mean, self.minibatch_riv, self.ret_cta,
                                  self.momentum,
                                  self.eps, self.fuse_relu, self.training, self.bn_group, self.my_data, self.pair_data, (self.magic), self.pair_data2, self.pair_data3,
                                  self.fwd_occupancy, self.fwd_grid_dim_x,
                                  self.bwd_occupancy, self.bwd_grid_dim_x,
                                  self.multi_stream)

    def __del__(self):
        if self.bn_group>1:
          bnp.close_remote_data(self.pair_handle)
          if self.bn_group>2:
              bnp.close_remote_data(self.pair_handle2)
              if self.bn_group>4:
                 bnp.close_remote_data(self.pair_handle3)
