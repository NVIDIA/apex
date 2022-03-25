import torch
import torch.distributed as dist
from torch import nn
import nccl_p2p as inc

# Communication free halo exchanger.
# NB! This halo exchanger does not exchange halos with neighbors as it should, it merely swaps the inputs
# NB! This is only useful for performance testing.
# NB! Do not use for actual production runs
class HaloExchanger(object):
    def __init__(self):
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()

class HaloExchangerNoComm(HaloExchanger):
    def __init__(self, world_size, spatial_group_size, rank, comm):
        super(HaloExchangerNoComm, self).__init__()

    def left_right_halo_exchange(self, left_output_halo, right_output_halo):
        return right_output_halo, left_output_halo

class HaloExchangerAllGather(HaloExchanger):
    def __init__(self, world_size, spatial_group_size, rank, comm):
        super(HaloExchangerAllGather, self).__init__()
        self.spatial_group_size = spatial_group_size
        self.local_rank = rank % spatial_group_size
        self.comm = comm

    def left_right_halo_exchange(self, left_output_halo, right_output_halo):
        N,Hh,W,C = list(left_output_halo.shape)
        send_halos = torch.empty((N,2*Hh,W,C),dtype=left_output_halo.dtype,device=left_output_halo.device)
        send_halos[:,:Hh,:,:].copy_(left_output_halo)
        send_halos[:,Hh:,:,:].copy_(right_output_halo)
        all_halos = torch.empty((N,2*Hh*self.spatial_group_size,W,C),dtype=left_output_halo.dtype,device=left_output_halo.device)
        all_halos = [all_halos[:,i*2*Hh:(i+1)*2*Hh,:,:] for i in range(self.spatial_group_size)]
        torch.distributed.all_gather(all_halos,send_halos,group=self.comm,no_copy=True)
        left_input_halo = all_halos[(self.spatial_group_size+self.local_rank-1)%self.spatial_group_size][:,Hh:,:,:]
        right_input_halo = all_halos[(self.local_rank+1)%self.spatial_group_size][:,:Hh,:,:]
        return left_input_halo, right_input_halo

class HaloExchangerSendRecv(HaloExchanger):
    def __init__(self, world_size, spatial_group_size, rank, comm):
        super(HaloExchangerSendRecv, self).__init__()
        self.world_size = world_size
        self.spatial_group_size = spatial_group_size
        nccl_id = inc.get_unique_nccl_id(1).cuda()
        torch.distributed.broadcast(nccl_id, 0)
        nccl_id = nccl_id.cpu()
        self.handle = inc.init_nccl_comm(nccl_id, rank, world_size)

    def left_right_halo_exchange(self, left_output_halo, right_output_halo):
        left_input_halo, right_input_halo = inc.left_right_halo_exchange(self.handle, left_output_halo, right_output_halo, self.spatial_group_size)
        return left_input_halo, right_input_halo
