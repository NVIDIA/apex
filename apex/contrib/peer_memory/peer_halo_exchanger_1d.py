import torch
from apex.contrib.peer_memory import PeerMemoryPool
import peer_memory_cuda as pm

class PeerHaloExchanger1d:
    def __init__(self, rank, peer_group_size, peer_pool, half_halo):
        self.peer_group_size = peer_group_size
        self.peer_rank = rank % peer_group_size
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
                top_out_halo = y[:,self.half_halo:2*self.half_halo,:,:]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, False, True)
                top_inp_halo = y[:,:self.half_halo,:,:]
                btm_out_halo = y[:,H:H+self.half_halo,:,:]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, False, True)
                btm_inp_halo = y[:,H+self.half_halo:H+2*self.half_halo,:,:]
            else:
                _, _, Hs, _ = list(y.shape)
                H = Hs - 2*self.half_halo
                top_out_halo = y[:,:,self.half_halo:2*self.half_halo,:]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, channels_last, True)
                top_inp_halo = y[:,:,:self.half_halo,:]
                btm_out_halo = y[:,:,H:H+self.half_halo,:]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, channels_last, True)
                btm_inp_halo = y[:,:,H+self.half_halo:H+2*self.half_halo,:]
        else:
            if explicit_nhwc:
                _, _, Ws, _ = list(y.shape)
                W = Ws - 2*self.half_halo
                top_out_halo = y[:,:,self.half_halo:2*self.half_halo,:]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, False, True)
                top_inp_halo = y[:,:,:self.half_halo,:]
                btm_out_halo = y[:,:,W:W+self.half_halo,:]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, False, True)
                btm_inp_halo = y[:,:,W+self.half_halo:W+2*self.half_halo,:]
            else:
                _, _, _, Ws = list(y.shape)
                W = Ws - 2*self.half_halo
                top_out_halo = y[:,:,:,self.half_halo:2*self.half_halo]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, channels_last, True)
                top_inp_halo = y[:,:,:,:self.half_halo]
                btm_out_halo = y[:,:,:,W:W+self.half_halo]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, channels_last, True)
                btm_inp_halo = y[:,:,:,W+self.half_halo:W+2*self.half_halo]
        top_neighbor = (self.peer_rank + self.peer_group_size - 1) % self.peer_group_size
        btm_neighbor = (self.peer_rank + 1) % self.peer_group_size
        pm.push_pull_halos_1d(
                diagnostics, explicit_nhwc, numSM,
                top_out_halo, top_tx[self.peer_rank], btm_tx[top_neighbor], top_inp_halo, 
                btm_out_halo, btm_tx[self.peer_rank], top_tx[btm_neighbor], btm_inp_halo,
                self.signals[top_neighbor], self.signals[btm_neighbor], self.signals[self.peer_rank]
                )
