import torch
from apex.contrib.peer_memory import PeerMemoryPool
import peer_memory_cuda as pm

class PeerHaloExchanger1d:
    def __init__(self, ranks, rank_in_group, peer_pool, half_halo):
        self.peer_group_size = len(ranks)
        self.ranks = ranks
        self.peer_rank = rank_in_group
        self.low_neighbor = (self.peer_rank + self.peer_group_size - 1) % self.peer_group_size
        self.high_neighbor = (self.peer_rank + 1) % self.peer_group_size
        self.low_zero = True if self.peer_rank == 0 else False
        self.high_zero = True if self.peer_rank == self.peer_group_size - 1 else False

        self.peer_pool = peer_pool
        self.signals = peer_pool.allocate_peer_tensors([2,4], torch.int32, False, False)
        self.signals[self.peer_rank].zero_()
        self.half_halo = half_halo

    def __call__(self, y, H_split=True, explicit_nhwc=False, numSM=1, diagnostics=False):
        channels_last = y.is_contiguous(memory_format=torch.channels_last) and not explicit_nhwc
        if H_split:
            if explicit_nhwc:
                _, Hs, _, _ = list(y.shape)
                H = Hs - 2*self.half_halo
                low_out_halo = y[:,self.half_halo:2*self.half_halo,:,:]
                low_tx = self.peer_pool.allocate_peer_tensors(list(low_out_halo.shape), low_out_halo.dtype, False, True)
                low_inp_halo = y[:,:self.half_halo,:,:]
                high_out_halo = y[:,H:H+self.half_halo,:,:]
                high_tx = self.peer_pool.allocate_peer_tensors(list(high_out_halo.shape), high_out_halo.dtype, False, True)
                high_inp_halo = y[:,H+self.half_halo:H+2*self.half_halo,:,:]
            else:
                _, _, Hs, _ = list(y.shape)
                H = Hs - 2*self.half_halo
                low_out_halo = y[:,:,self.half_halo:2*self.half_halo,:]
                low_tx = self.peer_pool.allocate_peer_tensors(list(low_out_halo.shape), low_out_halo.dtype, channels_last, True)
                low_inp_halo = y[:,:,:self.half_halo,:]
                high_out_halo = y[:,:,H:H+self.half_halo,:]
                high_tx = self.peer_pool.allocate_peer_tensors(list(high_out_halo.shape), high_out_halo.dtype, channels_last, True)
                high_inp_halo = y[:,:,H+self.half_halo:H+2*self.half_halo,:]
        else:
            if explicit_nhwc:
                _, _, Ws, _ = list(y.shape)
                W = Ws - 2*self.half_halo
                low_out_halo = y[:,:,self.half_halo:2*self.half_halo,:]
                low_tx = self.peer_pool.allocate_peer_tensors(list(low_out_halo.shape), low_out_halo.dtype, False, True)
                low_inp_halo = y[:,:,:self.half_halo,:]
                high_out_halo = y[:,:,W:W+self.half_halo,:]
                high_tx = self.peer_pool.allocate_peer_tensors(list(high_out_halo.shape), high_out_halo.dtype, False, True)
                high_inp_halo = y[:,:,W+self.half_halo:W+2*self.half_halo,:]
            else:
                _, _, _, Ws = list(y.shape)
                W = Ws - 2*self.half_halo
                low_out_halo = y[:,:,:,self.half_halo:2*self.half_halo]
                low_tx = self.peer_pool.allocate_peer_tensors(list(low_out_halo.shape), low_out_halo.dtype, channels_last, True)
                low_inp_halo = y[:,:,:,:self.half_halo]
                high_out_halo = y[:,:,:,W:W+self.half_halo]
                high_tx = self.peer_pool.allocate_peer_tensors(list(high_out_halo.shape), high_out_halo.dtype, channels_last, True)
                high_inp_halo = y[:,:,:,W+self.half_halo:W+2*self.half_halo]
        pm.push_pull_halos_1d(
                diagnostics, explicit_nhwc, numSM,
                self.low_zero, low_out_halo, low_tx[self.peer_rank], high_tx[self.low_neighbor], low_inp_halo, 
                self.high_zero, high_out_halo, high_tx[self.peer_rank], low_tx[self.high_neighbor], high_inp_halo,
                self.signals[self.low_neighbor], self.signals[self.high_neighbor], self.signals[self.peer_rank]
                )
