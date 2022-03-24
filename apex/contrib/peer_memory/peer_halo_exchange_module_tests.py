import torch
from apex.contrib.peer_memory import PeerMemoryPool
import peer_memory as pm


class HaloExchangerPeerMemory:
    def __init__(self, rank, peer_group_size, peer_pool):
        self.peer_group_size = peer_group_size
        self.peer_rank = rank % peer_group_size
        self.peer_pool = peer_pool
        self.signals = peer_pool.allocate_peer_tensors([2,4], torch.int32, False, False)
        self.signals[self.peer_rank].zero_()

    def __call__(self, y, half_halo, H_split=True, explicit_nhwc=False, numSM=1):
        channels_last = y.is_contiguous(memory_format=torch.channels_last)
        if H_split:
            if explicit_nhwc:
                _, Hs, _, _ = list(y.shape)
                H = Hs - 2*half_halo
                top_out_halo = y[:,half_halo:2*half_halo,:,:]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, False, True)
                top_inp_halo = y[:,:half_halo,:,:]
                btm_out_halo = y[:,H:H+half_halo,:,:]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, False, True)
                btm_inp_halo = y[:,H+half_halo:H+2*half_halo,:,:]
            else:
                _, _, Hs, _ = list(y.shape)
                H = Hs - 2*half_halo
                top_out_halo = y[:,:,half_halo:2*half_halo,:]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, channels_last, True)
                top_inp_halo = y[:,:,:half_halo,:]
                btm_out_halo = y[:,:,H:H+half_halo,:]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, channels_last, True)
                btm_inp_halo = y[:,:,H+half_halo:H+2*half_halo,:]
        else:
            if explicit_nhwc:
                _, _, Ws, _ = list(y.shape)
                W = Ws - 2*half_halo
                top_out_halo = y[:,:,half_halo:2*half_halo,:]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, False, True)
                top_inp_halo = y[:,:,:half_halo,:]
                btm_out_halo = y[:,:,W:W+half_halo,:]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, False, True)
                btm_inp_halo = y[:,:,W+half_halo:W+2*half_halo,:]
            else:
                _, _, _, Ws = list(y.shape)
                W = Ws - 2*half_halo
                top_out_halo = y[:,:,:,half_halo:2*half_halo]
                top_tx = self.peer_pool.allocate_peer_tensors(list(top_out_halo.shape), top_out_halo.dtype, channels_last, True)
                top_inp_halo = y[:,:,:,:half_halo]
                btm_out_halo = y[:,:,:,W:W+half_halo]
                btm_tx = self.peer_pool.allocate_peer_tensors(list(btm_out_halo.shape), btm_out_halo.dtype, channels_last, True)
                btm_inp_halo = y[:,:,:,W+half_halo:W+2*half_halo]
        top_neighbor = (self.peer_rank + self.peer_group_size - 1) % self.peer_group_size
        btm_neighbor = (self.peer_rank + 1) % self.peer_group_size
        pm.push_pull_halos_1d(
                False, #True if self.peer_rank == 0 else False,
                explicit_nhwc, numSM,
                top_out_halo, top_tx[self.peer_rank], btm_tx[top_neighbor], top_inp_halo, 
                btm_out_halo, btm_tx[self.peer_rank], top_tx[btm_neighbor], btm_inp_halo,
                self.signals[top_neighbor], self.signals[btm_neighbor], self.signals[self.peer_rank]
                )


def nccl_halo_ex(peer_rank, peer_group_size, y, half_halo, explicit_nhwc, H_split):
    if explicit_nhwc:
        if H_split:
            _, Hp, _, _ = list(y.shape)
            H = Hp - 2*half_halo
            top_out_halo = y[:,half_halo:2*half_halo,:,:]
            top_inp_halo = y[:,:half_halo,:,:]
            btm_out_halo = y[:,H:H+half_halo,:,:]
            btm_inp_halo = y[:,H+half_halo:H+2*half_halo,:,:]
        else:
            _, _, Wp, _ = list(y.shape)
            W = Wp - 2*half_halo
            top_out_halo = y[:,:,half_halo:2*half_halo,:]
            top_inp_halo = y[:,:,:half_halo,:]
            btm_out_halo = y[:,:,W:W+half_halo,:]
            btm_inp_halo = y[:,:,W+half_halo:W+2*half_halo,:]
    else:
        if H_split:
            _, _, Hp, _ = list(y.shape)
            H = Hp - 2*half_halo
            top_out_halo = y[:,:,half_halo:2*half_halo,:]
            top_inp_halo = y[:,:,:half_halo,:]
            btm_out_halo = y[:,:,H:H+half_halo,:]
            btm_inp_halo = y[:,:,H+half_halo:H+2*half_halo,:]
        else:
            _, _, _, Wp = list(y.shape)
            W = Wp - 2*half_halo
            top_out_halo = y[:,:,:,half_halo:2*half_halo]
            top_inp_halo = y[:,:,:,:half_halo]
            btm_out_halo = y[:,:,:,W:W+half_halo]
            btm_inp_halo = y[:,:,:,W+half_halo:W+2*half_halo]

    top_out_halo = top_out_halo.clone(memory_format=torch.preserve_format)
    btm_out_halo = btm_out_halo.clone(memory_format=torch.preserve_format)

    top_inp_halos = [torch.empty_like(top_out_halo) for _ in range(peer_group_size)]
    torch.distributed.all_gather(top_inp_halos, top_out_halo)
    btm_inp_halos = [torch.empty_like(btm_out_halo) for _ in range(peer_group_size)]
    torch.distributed.all_gather(btm_inp_halos, btm_out_halo)
    top_rank = (peer_rank + peer_group_size - 1) % peer_group_size
    btm_rank = (peer_rank + 1) % peer_group_size
    top_inp_halo.copy_(btm_inp_halos[top_rank])
    btm_inp_halo.copy_(top_inp_halos[btm_rank])


def single_test(peer_rank, peer_group_size, halo_ex, C, H, W, half_halo, dtype, memory_format, H_split, numSM=1):
    if memory_format == 1:
        # 1 -> explicit nhwc
        explicit_nhwc = True
        if H_split:
            y = torch.randn([1,H+2*half_halo,W,C], dtype=dtype, device='cuda')
            ym = y[:,half_halo:H+half_halo,:,:]
        else:
            y = torch.randn([1,H,W+2*half_halo,C], dtype=dtype, device='cuda')
            ym = y[:,:,half_halo:W+half_halo,:]
    else:
        # 2 -> native nhwc
        # 3 -> nchw
        explicit_nhwc = False
        if H_split:
            y = torch.randn([1,C,H+2*half_halo,W], dtype=dtype, device='cuda')
            if memory_format == 2:
                y = y.to(memory_format=torch.channels_last)
            ym = y[:,:,half_halo:H+half_halo,:]
        else:
            y = torch.randn([1,C,H,W+2*half_halo], dtype=dtype, device='cuda')
            if memory_format == 2:
                y = y.to(memory_format=torch.channels_last)
            ym = y[:,:,:,half_halo:W+half_halo]
    y2 = y.clone()
    halo_ex(y, half_halo, H_split, explicit_nhwc, numSM)
    nccl_halo_ex(peer_rank, peer_group_size, y2, half_halo, explicit_nhwc, H_split)
    is_equal = torch.all(torch.eq(y,y2))
    if peer_rank == 0:
        if memory_format == 1:
            memory_format_str = "explicit_nhwc"
        elif memory_format == 2:
            memory_format_str = "native nhwc"
        elif memory_format == 3:
            memory_format_str = "nchw"
        else:
            memory_format_str = "???"
        if is_equal:
            print("SUCCESS : N,C,H,W = 1,%d,%d,%d, half_halo=%d, %s, %s, %s" % (C,H,W,half_halo,str(dtype),memory_format_str,"H-split" if H_split else "W-split"))
        else:
            print("FAILURE : N,C,H,W = 1,%d,%d,%d, half_halo=%d, %s, %s, %s" % (C,H,W,half_halo,str(dtype),memory_format_str,"H-split" if H_split else "W-split"))

    # peer memory flag sync relies on there being at least one barrier per step
    torch.distributed.barrier()


def H_split_tests(N, C, H, W, half_halo, rank, world_size, halo_ex):
    Hr = 8*world_size
    Hp = ((H + Hr - 1) // Hr) * 8

    for i in range(4):
        div = int(pow(2,i))
        single_test(rank, world_size, halo_ex, C*div, Hp//div, W//div, half_halo, torch.float16, 1, True)
        single_test(rank, world_size, halo_ex, C*div, Hp//div, W//div, half_halo, torch.float16, 2, True)
        single_test(rank, world_size, halo_ex, C*div, Hp//div, W//div, half_halo, torch.float16, 3, True)


def W_split_tests(N, C, H, W, half_halo, rank, world_size, halo_ex):
    Wr = 8*world_size
    Wp = ((W + Wr - 1) // Wr) * 8

    for i in range(4):
        div = int(pow(2,i))
        single_test(rank, world_size, halo_ex, C*div, H//div, Wp//div, half_halo, torch.float16, 1, False)
        single_test(rank, world_size, halo_ex, C*div, H//div, Wp//div, half_halo, torch.float16, 2, False)
        single_test(rank, world_size, halo_ex, C*div, H//div, Wp//div, half_halo, torch.float16, 3, False)


def main():
    # for this trivial example peer_rank == rank and peer_group_size == world_size

    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)
    pool = PeerMemoryPool(rank, world_size, world_size, 64*1024, 2*1024*1024)

    halo_ex = HaloExchangerPeerMemory(rank, world_size, pool)

    half_halo = 1

    H_split_tests(1,64,336,200, half_halo,rank,world_size,halo_ex)
    W_split_tests(1,64,200,336, half_halo,rank,world_size,halo_ex)


if __name__ == "__main__":
    main()
