import torch
import numpy as np
import peer_memory_cuda as pm

class PeerMemoryPool(object):

    def __init__(self, rank, world_size, peer_group_size, static_size, dynamic_size):
        self.peer_group = rank // peer_group_size
        self.peer_rank = rank % peer_group_size
        self.peer_group_size = peer_group_size
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
        self.peer_raw = pm.get_raw_peers(peer_raw_ipcs, self.peer_rank, self.raw)
        self.static_offset = 0
        self.dynamic_offset = 0

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
