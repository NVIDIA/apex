import torch
import numpy as np
import peer_memory

class PeerMemoryPool(object):

    def __init__(self, rank, world_size, peer_group_size, size):
        self.peer_group = rank // peer_group_size
        self.peer_rank = rank % peer_group_size
        self.peer_group_size = peer_group_size
        self.alignment = 256
        self.size = size
        # allocate giant pool of device memory
        self.raw = allocate_raw(size)
        # exchange peer pointers with nccl
        raw_ipc = get_raw_ipc_address(self.raw).cuda()
        peer_raw_ipcs = [torch.empty_like(raw_ipc) for _ in range(world_size)]
        torch.distributed.all_gather(peer_raw_ipcs, raw_ipc)
        peer_raw_ipcs = torch.stack(peer_raw_ipcs).cpu()
        self.peer_raw = get_raw_peers(peer_raw_ipcs, self.peer_rank, self.raw)
        self.current = 0

    def __del__(self):
        free_raw(self.raw)

    def reset(self):
        self.current = 0

    def allocate_peer_tensors(self, shape, dtype):
        nels = np.prod(shape)
        if dtype == torch.float16:
            elem_size = 2
            start = ((self.current + self.alignment - 1) // self.alignment) * self.alignment
            self.current = start + nels * elem_size
            assert(self.current < self.size), "Peer memory pool exhausted"
            return [blob_view_half(pr + start, shape) for pr in self.peer_raw]
        elif dtype == torch.float32:
            elem_size = 4
            start = ((self.current + self.alignment - 1) // self.alignment) * self.alignment
            self.current = start + nels * elem_size
            assert(self.current < self.size), "Peer memory pool exhausted"
            return [blob_view_float(pr + start, shape) for pr in self.peer_raw]
        elif dtype == torch.int32:
            elem_size = 4
            start = ((self.current + self.alignment - 1) // self.alignment) * self.alignment
            self.current = start + nels * elem_size
            assert(self.current < self.size), "Peer memory pool exhausted"
            return [blob_view_int(pr + start, shape) for pr in self.peer_raw]
        else:
            assert(False), "Unknown dtype : %s" % (str(dtype))
