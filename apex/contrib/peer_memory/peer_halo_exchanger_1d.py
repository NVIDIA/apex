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
        self.half_halo = half_halo

    def _allocate_peer_tensor(self, halo):

        # Compute size in bytes
        # Note: Pad buffer so each CUDA block gets required buffer size
        size = 4 * halo.numel() * halo.element_size()
        size_per_block = 128 * 2 * 16   # 128 threads each require two 128b buffers
        size = (size + size_per_block - 1) // size_per_block * size_per_block

        # Construct dtype peer buffer with desired size
        shape = [1, 1, 1, size // halo.element_size()]
        return self.peer_pool.allocate_peer_tensors(shape, halo.dtype, False, True)

    def __call__(self, y, H_split=True, explicit_nhwc=False, numSM=0, diagnostics=False):
        channels_last = y.is_contiguous(memory_format=torch.channels_last) and not explicit_nhwc
        if H_split:
            if explicit_nhwc:
                _, Hs, _, _ = list(y.shape)
                H = Hs - 2*self.half_halo
                low_out_halo = y[:,self.half_halo:2*self.half_halo,:,:]
                low_tx = self._allocate_peer_tensor(low_out_halo)
                low_inp_halo = y[:,:self.half_halo,:,:]
                high_out_halo = y[:,H:H+self.half_halo,:,:]
                high_tx = self._allocate_peer_tensor(high_out_halo)
                high_inp_halo = y[:,H+self.half_halo:H+2*self.half_halo,:,:]
            else:
                _, _, Hs, _ = list(y.shape)
                H = Hs - 2*self.half_halo
                low_out_halo = y[:,:,self.half_halo:2*self.half_halo,:]
                low_tx = self._allocate_peer_tensor(low_out_halo)
                low_inp_halo = y[:,:,:self.half_halo,:]
                high_out_halo = y[:,:,H:H+self.half_halo,:]
                high_tx = self._allocate_peer_tensor(high_out_halo)
                high_inp_halo = y[:,:,H+self.half_halo:H+2*self.half_halo,:]
        else:
            if explicit_nhwc:
                _, _, Ws, _ = list(y.shape)
                W = Ws - 2*self.half_halo
                low_out_halo = y[:,:,self.half_halo:2*self.half_halo,:]
                low_tx = self._allocate_peer_tensor(low_out_halo)
                low_inp_halo = y[:,:,:self.half_halo,:]
                high_out_halo = y[:,:,W:W+self.half_halo,:]
                high_tx = self._allocate_peer_tensor(high_out_halo)
                high_inp_halo = y[:,:,W+self.half_halo:W+2*self.half_halo,:]
            else:
                _, _, _, Ws = list(y.shape)
                W = Ws - 2*self.half_halo
                low_out_halo = y[:,:,:,self.half_halo:2*self.half_halo]
                low_tx = self._allocate_peer_tensor(low_out_halo)
                low_inp_halo = y[:,:,:,:self.half_halo]
                high_out_halo = y[:,:,:,W:W+self.half_halo]
                high_tx = self._allocate_peer_tensor(high_out_halo)
                high_inp_halo = y[:,:,:,W+self.half_halo:W+2*self.half_halo]
        pm.push_pull_halos_1d(
                diagnostics, explicit_nhwc, numSM, self.peer_rank,
                self.low_zero, low_out_halo, low_tx[self.peer_rank], high_tx[self.low_neighbor], low_inp_halo,
                self.high_zero, high_out_halo, high_tx[self.peer_rank], low_tx[self.high_neighbor], high_inp_halo,
                )
