import torch
from apex.contrib.peer_memory import PeerMemoryPool, PeerHaloExchanger1d
import peer_memory_cuda as pm

# How to run:
# torchrun --nproc_per_node <num-GPU> <this-python-prog>
# <num-GPU> must be a power of 2 greater than 1.


# Output of this function is used as ground truth in module tests.
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


def single_test(peer_rank, peer_group_size, halo_ex, C, H, W, half_halo, dtype, memory_format, H_split, num_steps, numSM=1):
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
    y3 = y.clone()
    list_y = []
    for step in range(num_steps):
        halo_ex(y, H_split, explicit_nhwc, numSM)
        list_y.append(y.clone())
        y.copy_(y3)
        halo_ex.peer_pool.reset()
        torch.distributed.barrier()
    y2 = y3.clone()
    list_y2 = []
    for step in range(num_steps):
        nccl_halo_ex(peer_rank, peer_group_size, y2, half_halo, explicit_nhwc, H_split)
        list_y2.append(y2.clone())
        y2.copy_(y3)
    is_equal = [torch.all(torch.eq(yy,yy2)) for yy,yy2 in zip(list_y,list_y2)]
    is_equal = torch.tensor(is_equal, dtype=torch.bool)
    is_equal = torch.all(is_equal)
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


def H_split_tests(N, C, H, W, half_halo, rank, world_size, halo_ex, num_steps):
    Hr = 8*world_size
    Hp = ((H + Hr - 1) // Hr) * 8

    for i in range(4):
        div = int(pow(2,i))
        single_test(rank, world_size, halo_ex, C*div, Hp//div, W//div, half_halo, torch.float16, 1, True, num_steps)
        single_test(rank, world_size, halo_ex, C*div, Hp//div, W//div, half_halo, torch.float16, 2, True, num_steps)
        single_test(rank, world_size, halo_ex, C*div, Hp//div, W//div, half_halo, torch.float16, 3, True, num_steps)


def W_split_tests(N, C, H, W, half_halo, rank, world_size, halo_ex, num_steps):
    Wr = 8*world_size
    Wp = ((W + Wr - 1) // Wr) * 8

    for i in range(4):
        div = int(pow(2,i))
        single_test(rank, world_size, halo_ex, C*div, H//div, Wp//div, half_halo, torch.float16, 1, False, num_steps)
        single_test(rank, world_size, halo_ex, C*div, H//div, Wp//div, half_halo, torch.float16, 2, False, num_steps)
        single_test(rank, world_size, halo_ex, C*div, H//div, Wp//div, half_halo, torch.float16, 3, False, num_steps)


def main():
    # for this trivial example peer_rank == rank and peer_group_size == world_size

    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)
    pool = PeerMemoryPool(rank, world_size, world_size, 64*1024, 2*1024*1024)

    num_steps = 100

    half_halo = 1
    halo_ex = PeerHaloExchanger1d(rank, world_size, pool, half_halo)

    H_split_tests(1,64,336,200, half_halo,rank,world_size,halo_ex,num_steps)
    W_split_tests(1,64,200,336, half_halo,rank,world_size,halo_ex,num_steps)


if __name__ == "__main__":
    main()
