import torch
import torch.distributed as dist
from torch import nn
import nccl_p2p_cuda as inc
import peer_memory_cuda as pm

# Communication free halo exchanger.
# NB! This halo exchanger does not exchange halos with neighbors as it should, it merely swaps the inputs
# NB! This is only useful for performance testing.
# NB! Do not use for actual production runs
class HaloExchanger(object):
    def __init__(self, ranks, rank_in_group):
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        self.stream3 = torch.cuda.Stream()
        self.group_size = len(ranks)
        self.ranks = ranks
        self.rank_in_group = rank_in_group
        self.wrap_around_left_rank_in_group = (rank_in_group + self.group_size - 1) % self.group_size
        self.wrap_around_right_rank_in_group = (rank_in_group + 1) % self.group_size
        self.left_rank = ranks[rank_in_group-1] if rank_in_group > 0 else -1
        self.left_zero = True if rank_in_group == 0 else False
        self.right_rank = ranks[rank_in_group+1] if rank_in_group < self.group_size - 1 else -1
        self.right_zero = True if rank_in_group == self.group_size - 1 else False

class HaloExchangerNoComm(HaloExchanger):
    def __init__(self, ranks, rank_in_group):
        super(HaloExchangerNoComm, self).__init__(ranks, rank_in_group)

    def left_right_halo_exchange(self, left_output_halo, right_output_halo, left_input_halo=None, right_input_halo=None):
        if left_input_halo is None:
            return right_output_halo, left_output_halo
        else:
            left_input_halo.copy_(right_output_halo)
            right_input_halo.copy_(left_output_halo)

class HaloExchangerAllGather(HaloExchanger):
    def __init__(self, ranks, rank_in_group, comm):
        super(HaloExchangerAllGather, self).__init__(ranks, rank_in_group)
        # self.comm must be NCCL process_group created with torch.distributed.new_group(ranks=ranks)
        self.comm = comm

    def left_right_halo_exchange(self, left_output_halo, right_output_halo, left_input_halo=None, right_input_halo=None):
        N,Hh,W,C = list(left_output_halo.shape)
        send_halos = torch.empty((N,2*Hh,W,C),dtype=left_output_halo.dtype,device=left_output_halo.device)
        send_halos[:,:Hh,:,:].copy_(left_output_halo)
        send_halos[:,Hh:,:,:].copy_(right_output_halo)
        all_halos = torch.empty((N,2*Hh*self.group_size,W,C),dtype=left_output_halo.dtype,device=left_output_halo.device)
        all_halos = [all_halos[:,i*2*Hh:(i+1)*2*Hh,:,:] for i in range(self.group_size)]
        torch.distributed.all_gather(all_halos,send_halos,group=self.comm,no_copy=True)
        ag_left_input_halo = all_halos[self.wrap_around_left_rank_in_group][:,Hh:,:,:]
        ag_right_input_halo = all_halos[self.wrap_around_right_rank_in_group][:,:Hh,:,:]
        if left_input_halo is None:
            if self.left_zero:
                ag_left_input_halo.zero_()
            if self.right_zero:
                ag_right_input_halo.zero_()
            return ag_left_input_halo, ag_right_input_halo
        else:
            if self.left_zero:
                left_input_halo.zero_()
            else:
                left_input_halo.copy_(ag_left_input_halo)
            if self.right_zero:
                right_input_halo.zero_()
            else:
                right_input_halo.copy_(ag_right_input_halo)

class HaloExchangerSendRecv(HaloExchanger):
    def __init__(self, ranks, rank_in_group):
        super(HaloExchangerSendRecv, self).__init__(ranks, rank_in_group)
        nccl_id = inc.get_unique_nccl_id(1).cuda()
        torch.distributed.broadcast(nccl_id, 0)
        nccl_id = nccl_id.cpu()
        print("%d :: nccl_id = %s" % (torch.distributed.get_rank(), str(nccl_id)))
        # Create another global nccl communicator in addition to the one created by torch.distributed.init_process_group("nccl")
        # This is unavoidable because the underlying NCCL communicator torch.distributed creates is a protected variable, hence
        # it cannot be accessed from another class.
        # TODO: Figure out a way to avoid creating a second global communicator
        assert(torch.distributed.get_rank() == self.ranks[self.rank_in_group]), "ranks[%d](%d) != torch.distributed.get_rank()(%d)" % (self.rank_in_group, self.ranks[self.rank_in_group], torch.distributed.get_rank())
        self.handle = inc.init_nccl_comm(nccl_id, torch.distributed.get_rank(), torch.distributed.get_world_size())

    def left_right_halo_exchange(self, left_output_halo, right_output_halo, left_input_halo=None, right_input_halo=None):
        if left_input_halo is None:
            left_input_halo, right_input_halo = inc.left_right_halo_exchange(self.handle, self.left_rank, self.right_rank , left_output_halo, right_output_halo)
            return left_input_halo, right_input_halo
        else:
            inc.left_right_halo_exchange_inplace(self.handle, self.left_rank, self.right_rank, left_output_halo, right_output_halo, left_input_halo, right_input_halo)

class HaloExchangerPeer(HaloExchanger):
    def __init__(self, ranks, rank_in_group, peer_pool, explicit_nhwc, numSM=1):
        super(HaloExchangerPeer, self).__init__(ranks, rank_in_group)
        self.diagnostics = False
        self.explicit_nhwc = explicit_nhwc
        self.numSM = numSM
        self.peer_pool = peer_pool
        self.signals = peer_pool.allocate_peer_tensors([2,4], torch.int32, False, False)
        self.signals[self.rank_in_group].zero_()

    def left_right_halo_exchange(self, left_output_halo, right_output_halo, left_input_halo=None, right_input_halo=None):
        inplace = False if left_input_halo is None and right_input_halo is None else True
        if not inplace:
            left_input_halo = torch.empty_like(right_output_halo)
            right_input_halo = torch.empty_like(left_output_halo)
        channels_last = left_output_halo.is_contiguous(memory_format=torch.channels_last) and not self.explicit_nhwc
        left_tx = self.peer_pool.allocate_peer_tensors(list(left_output_halo.shape), left_output_halo.dtype, channels_last, True)
        right_tx = self.peer_pool.allocate_peer_tensors(list(right_output_halo.shape), right_output_halo.dtype, channels_last, True)
        pm.push_pull_halos_1d(
                self.diagnostics, self.explicit_nhwc, self.numSM,
                left_output_halo,  left_tx[self.rank_in_group],  right_tx[self.wrap_around_left_rank_in_group], left_input_halo,
                right_output_halo, right_tx[self.rank_in_group], left_tx[self.wrap_around_right_rank_in_group],  right_input_halo,
                self.signals[self.wrap_around_left_rank_in_group], self.signals[self.wrap_around_right_rank_in_group], self.signals[self.rank_in_group]
                )
        # TODO: Add to push_pull_halos_1d kernel
        if self.left_zero:
            left_input_halo.zero_()
        if self.right_zero:
            right_input_halo.zero_()
        if not inplace:
            return left_input_halo, right_input_halo

# Class that combines input volume with halos from neighbors (1d).
class HaloPadder:
    def __init__(self, halo_ex):
        self.halo_ex = halo_ex
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()

    def __call__(self, y, half_halo, explicit_nhwc, H_split):
        channels_last = not explicit_nhwc and y.is_contiguous(memory_format=torch.channels_last)
        if explicit_nhwc:
            N,H,W,C = list(y.shape)
            if H_split:
                padded_shape = [N,H+2*half_halo,W,C]
                ypad = torch.empty(shape=padded_shape, dtype=y.dtype, device=y.device, memory_format=torch.contiguous_format)
                yleft = ypad[:,:half_halo,:,:]
                ymid = ypad[:,half_halo:H+half_halo,:,:]
                yright = ypad[:,H+half_halo:H+2*half_halo,:,:]
                oleft = y[:,:half_halo,:,:]
                oright = y[:,H-half_halo:,:,:]
            else:
                padded_shape = [N,H,W+2*half_halo,C]
                ypad = torch.empty(shape=padded_shape, dtype=y.dtype, device=y.device, memory_format=torch.contiguous_format)
                yleft = ypad[:,:,:half_halo,:]
                ymid = ypad[:,:,half_halo:W+half_halo,:]
                yright = ypad[:,:,W+half_halo:W+2*half_halo,:]
                oleft = y[:,:,:half_halo,:]
                oright = y[:,:,W-half_halo:,:]
        else:
            N,C,H,W = list(y.shape)
            if H_split:
                padded_shape = [N,C,H+2*half_halo,W]
                ypad = torch.empty(shape=padded_shape, dtype=y.dtype, device=y.device, memory_format=torch.channels_last)
                yleft = ypad[:,:,:half_halo,:]
                ymid = ypad[:,:,half_halo:H+half_halo,:]
                yright = ypad[:,:,H+half_halo:H+2*half_halo,:]
                oleft = y[:,:,:half_halo,:]
                oright = y[:,:,H-half_halo:,:]
            else:
                padded_shape = [N,C,H,W+2*half_halo]
                ypad = torch.empty(shape=padded_shape, dtype=y.dtype, device=y.device, memory_format=torch.channels_last)
                yleft = ypad[:,:,:,:half_halo]
                ymid = ypad[:,:,:,half_halo:W+half_halo]
                yright = ypad[:,:,:,W+half_halo:W+2*half_halo]
                oleft = y[:,:,:,:half_halo]
                oright = y[:,:,:,W-half_halo:]
        with torch.cuda.stream(self.stream1):
            self.halo_ex(oleft, oright, yleft, yright)
        with torch.cuda.stream(self.stream2):
            ymid.copy_(y)
        return ypad

    def wait(self):
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.stream1)
        current_stream.wait_stream(self.stream2)
