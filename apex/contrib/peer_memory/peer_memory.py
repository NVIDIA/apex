import torch
import numpy as np
import peer_memory_cuda as pm

class PeerMemoryPool(object):

    def __init__(self, static_size, dynamic_size, peer_ranks=None):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        ngpus = min(torch.cuda.device_count(), world_size)
        peer_group_size = ngpus
        peer_group = rank // ngpus
        peer_rank_base = peer_group * ngpus
        peer_rank = rank - peer_rank_base
        if peer_ranks is None:
            peer_ranks = [i+peer_rank_base for i in range(peer_group_size)]
        peer_rank_start = peer_rank_base
        peer_rank_end = peer_rank_start + peer_group_size - 1
        for pr in peer_ranks:
            assert(pr >= peer_rank_start and pr <= peer_rank_end), "%d :: peer_rank %d not on same node (ranks=[%d,%d])" % (rank, pr, peer_rank_start, peer_rank_end)

        self.alignment = 256
        self.static_size = ((static_size + self.alignment - 1) // self.alignment) * self.alignment
        self.dynamic_size = ((dynamic_size + self.alignment - 1) // self.alignment) * self.alignment

        # allocate giant pool of device memory
        self.raw = pm.allocate_raw(self.static_size+self.dynamic_size)

        # exchange peer pointers with nccl
        raw_ipc = pm.get_raw_ipc_address(self.raw).cuda()
        peer_raw_ipcs = [torch.empty_like(raw_ipc) for _ in range(world_size)]
        torch.distributed.all_gather(peer_raw_ipcs, raw_ipc)
        peer_raw_ipcs = torch.stack(peer_raw_ipcs).cpu()

        # extract IPC pointers for ranks on same node
        peer_raw = pm.get_raw_peers(peer_raw_ipcs[peer_rank_base:peer_rank_base+ngpus], peer_rank, self.raw)
        self.peer_raw = [peer_raw[peer_rank-peer_rank_base] for peer_rank in peer_ranks]
        self.static_offset = 0
        self.dynamic_offset = 0
        self.peer_ranks = peer_ranks

    def __del__(self):
        pm.free_raw(self.raw)

    def reset(self):
        self.dynamic_offset = 0

    def allocate_peer_tensors(self, shape, dtype, channels_last, dynamic):
        nels = np.prod(shape)
        if dtype == torch.float16:
            elem_size = 2
            if dynamic:
                start = ((self.dynamic_offset + self.alignment - 1) // self.alignment) * self.alignment
                self.dynamic_offset =  start + nels * elem_size
                assert(self.dynamic_offset < self.dynamic_size), "Dynamic peer memory pool exhausted"
                return [pm.blob_view_half(pr + self.static_size + start, shape, channels_last) for pr in self.peer_raw]
            else:
                start = ((self.static_offset + self.alignment - 1) // self.alignment) * self.alignment
                self.static_offset = start + nels * elem_size
                assert(self.static_offset < self.static_size), "Static peer memory pool exhausted"
                return [pm.blob_view_half(pr + start, shape, channels_last) for pr in self.peer_raw]
        if dtype == torch.float32:
            elem_size = 4
            if dynamic:
                start = ((self.dynamic_offset + self.alignment - 1) // self.alignment) * self.alignment
                self.dynamic_offset =  start + nels * elem_size
                assert(self.dynamic_offset < self.dynamic_size), "Dynamic peer memory pool exhausted"
                return [pm.blob_view_float(pr + self.static_size + start, shape, channels_last) for pr in self.peer_raw]
            else:
                start = ((self.static_offset + self.alignment - 1) // self.alignment) * self.alignment
                self.static_offset = start + nels * elem_size
                assert(self.static_offset < self.static_size), "Static peer memory pool exhausted"
                return [pm.blob_view_float(pr + start, shape, channels_last) for pr in self.peer_raw]
        if dtype == torch.int32:
            elem_size = 4
            if dynamic:
                start = ((self.dynamic_offset + self.alignment - 1) // self.alignment) * self.alignment
                self.dynamic_offset =  start + nels * elem_size
                assert(self.dynamic_offset < self.dynamic_size), "Dynamic peer memory pool exhausted"
                return [pm.blob_view_int(pr + self.static_size + start, shape, channels_last) for pr in self.peer_raw]
            else:
                start = ((self.static_offset + self.alignment - 1) // self.alignment) * self.alignment
                self.static_offset = start + nels * elem_size
                assert(self.static_offset < self.static_size), "Static peer memory pool exhausted"
                return [pm.blob_view_int(pr + start, shape, channels_last) for pr in self.peer_raw]
        else:
            assert(False), "dtype %s not supported" % (str(dtype))
