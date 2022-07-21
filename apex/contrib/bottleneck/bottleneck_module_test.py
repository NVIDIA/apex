import torch
from apex.contrib.bottleneck import Bottleneck, SpatialBottleneck
from apex.contrib.bottleneck import HaloExchangerNoComm, HaloExchangerAllGather, HaloExchangerSendRecv, HaloExchangerPeer
from apex.contrib.peer_memory import PeerMemoryPool


def ground_truth_bottleneck(C, dtype, explicit_nhwc):
    bottleneck = Bottleneck(C,C,C,use_cudnn=True,explicit_nhwc=explicit_nhwc)
    bottleneck.to(dtype=dtype, device='cuda')
    for p in bottleneck.parameters():
        torch.distributed.broadcast(p, 0)
    for b in bottleneck.buffers():
        torch.distributed.broadcast(b, 0)
    return bottleneck


def print_bottleneck_p_and_b(bottleneck):
    with torch.no_grad():
        for n,p in bottleneck.named_parameters():
            print("%s :: %s" % (n, str(p.norm(p=2,dtype=torch.float32))))
        for n,p in bottleneck.named_buffers():
            print("%s :: %s" % (n, str(p.norm(p=2,dtype=torch.float32))))


def has_nan(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for xx in x:
            if torch.any(torch.isnan(xx)):
                return True
        return False
    elif isinstance(x, dict):
        for k,v in x.items():
            if torch.any(torch.isnan(v)):
                return True
    else:
        return torch.any(torch.isnan(x))


def rel_diff_t(xx1, xx2):
    return ((xx1 - xx2).norm(p=2,dtype=torch.float32) / (xx1 + xx2).norm(p=2,dtype=torch.float32)).item()


def rel_diff(x1, x2):
    if isinstance(x1, list) or isinstance(x1, tuple):
        return [rel_diff_t(xx1,xx2) for xx1,xx2 in zip(x1,x2)]
    elif isinstance(x1, dict):
        return [rel_diff_t(xx1, xx2) for (k1,xx1), (k2,xx2) in zip(x1.items(),x2.items())]
    else:
        return rel_diff_t(x1,x2)


def graph_it(bottleneck, x):
    print("Graphing")
    with torch.no_grad():
        x = x.clone()
        x.grad = None
        x.requires_grad = True
    return torch.cuda.make_graphed_callables(bottleneck, (x,))


def clone_inputs(bottleneck, x, dy=None):
    with torch.no_grad():
        x = x.clone()
        x.grad = None
        x.requires_grad = True
        if dy is None:
            y = bottleneck(x)
            dy = torch.randn_like(y) / 1e2
            torch.distributed.broadcast(dy, 0)
    return x, dy


def fprop_and_bprop(bottleneck, x, dy):
    y = bottleneck(x)
    y.backward(dy)
    dgrad = x.grad.detach()
    wgrad = {}
    for n,p in bottleneck.named_parameters():
        wgrad[n] = p.grad.detach()
    return x, y, dy, dgrad, wgrad


def ground_truth(N, C, H, W, dtype, memory_format, bottleneck):
    if memory_format == 1:
        # 1 -> explicit nhwc
        explicit_nhwc = True
        with torch.no_grad():
            x = torch.randn([N,H,W,C], dtype=dtype, device='cuda')
            torch.distributed.broadcast(x, 0)
            x, dy = clone_inputs(bottleneck, x)
        return fprop_and_bprop(bottleneck, x, dy)
    else:
        # 2 -> native nhwc
        # 3 -> nchw
        explicit_nhwc = False
        assert(False), "Not implemented yet"


def print_ground_truth(gt):
    x, y, dy, dgrad, wgrad = gt
    if has_nan(y) or has_nan(dgrad) or has_nan(wgrad):
        print("Error! Ground truth has NAN")
    else:
        print("Ok! No NAN found in ground truth")


def apply_to_different_bottleneck(gt, bottleneck):
    with torch.no_grad():
        x, _, dy, _, _ = gt
        x, dy = clone_inputs(bottleneck, x, dy)
    return fprop_and_bprop(bottleneck, x, dy)


def compare_single_field(results, f1, f2, l0, l1, l2):
    if has_nan(f1) and has_nan(f2):
        results[l0] = "both NAN"
    elif has_nan(f1):
        results[l0] = "%s.%s NAN" % (l1, l0)
    elif has_nan(f2):
        results[l0] = "%s.%s NAN" % (l2, l0)
    else:
        results[l0] = "%s" % (str(rel_diff(f1,f2)))


def compare(gt, bt):
    x1, y1, dy1, dgrad1, wgrad1 = gt
    x2, y2, dy2, dgrad2, wgrad2 = bt
    results = {}
    compare_single_field(results, y1, y2, "y", "gt", "bt")
    compare_single_field(results, dy1, dy2, "dy", "gt", "bt")
    compare_single_field(results, dgrad1, dgrad2, "dgrad", "gt", "bt")
    compare_single_field(results, wgrad1, wgrad2, "wgrad", "gt", "bt")
    for i in range(torch.distributed.get_world_size()):
        if i == torch.distributed.get_rank():
            print(i,results)
        torch.distributed.barrier()


def spatial_parallel_bottleneck(C, dtype, explicit_nhwc, gt_bottleneck, spatial_parallel_args):
    spatial_bottleneck = SpatialBottleneck(C,C,C,use_cudnn=True,explicit_nhwc=explicit_nhwc,spatial_parallel_args=spatial_parallel_args)
    spatial_bottleneck.to(dtype=dtype, device='cuda')
    with torch.no_grad():
        sp = {}
        for n,p in spatial_bottleneck.named_parameters():
            sp[n] = p
        for n,p in gt_bottleneck.named_parameters():
            sp[n].copy_(p)
        sb = {}
        for n,b in spatial_bottleneck.named_buffers():
            sb[n] = b
        for n,b in gt_bottleneck.named_buffers():
            sb[n].copy_(b)
    return spatial_bottleneck

def n_way_spatial(halex, gt_bottleneck, gt, explicit_nhwc, world_size, rank, fp32_reduce=False):
    assert(explicit_nhwc), "Only tested for explicit nhwc"

    x, _, dy, _, _ = gt
    N, H, W, C = list(x.shape) # Tensor is already shaped properly for n-way parallel
    dtype = x.dtype

    spatial_group_size = world_size
    spatial_group_rank = rank
    spatial_communicator = None
    spatial_halo_exchanger = halex
    spatial_method = 1 # 1 -> overlap halo and main conv, 2 -> wait for halo, conv on padded x
    use_delay_kernel = False
    spatial_parallel_args = (spatial_group_size, spatial_group_rank, spatial_communicator, spatial_halo_exchanger, spatial_method, use_delay_kernel)
    spatial_bottleneck = spatial_parallel_bottleneck(C, dtype, explicit_nhwc, gt_bottleneck, spatial_parallel_args)

    with torch.no_grad():
        Hs = H // spatial_group_size
        xs = x[:,spatial_group_rank*Hs:(spatial_group_rank+1)*Hs,:,:].clone()
        dys = dy[:,spatial_group_rank*Hs:(spatial_group_rank+1)*Hs,:,:].clone()
        xs.requires_grad = True

    spatial_bottleneck = graph_it(spatial_bottleneck, xs)
    _, y, _, dgrad, wgrad = fprop_and_bprop(spatial_bottleneck, xs, dys)

    # gather output pieces
    for n,p in wgrad.items():
        if fp32_reduce:
            p32 = p.float()
            torch.distributed.all_reduce(p32)
            p.copy_(p32.half())
        else:
            torch.distributed.all_reduce(p)
    ys = [torch.empty_like(y) for _ in range(spatial_group_size)]
    torch.distributed.all_gather(ys,y)
    y = torch.cat(ys,dim=1)
    dgrads = [torch.empty_like(dgrad) for _ in range(spatial_group_size)]
    torch.distributed.all_gather(dgrads,dgrad)
    dgrad = torch.cat(dgrads,dim=1)
    return x, y, dy, dgrad, wgrad


def main():
    torch.use_deterministic_algorithms(True)

    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)

    explicit_nhwc = True

    dtype = torch.float16
    N, C, H, W = 1, 64, 200, 336
    Hs = ((H+8*world_size-1) // (8*world_size)) * 8
    H = Hs*world_size
    gt_bottleneck = ground_truth_bottleneck(C, dtype, explicit_nhwc)
    gt = ground_truth(N, C, H, W, dtype, 1, gt_bottleneck)

    # verify that spatial bottleneck with group_size 1 produces same results as ground truth bottleneck
    spatial_bottleneck = spatial_parallel_bottleneck(C, dtype, explicit_nhwc, gt_bottleneck, None)
    bt = apply_to_different_bottleneck(gt, spatial_bottleneck)
    compare(gt, bt)
    #print_bottleneck_p_and_b(gt_bottleneck)
    #print_bottleneck_p_and_b(spatial_bottleneck)

    group_size = world_size
    group = rank // group_size
    ranks = [group*group_size+i for i in range(group_size)]
    rank_in_group = rank % group_size

    spatial_group_size = world_size
    spatial_communicator = None

    peer_pool = PeerMemoryPool(64*1024*1024, 2*1024*1024, ranks)

    #class HaloExchangerNoComm(HaloExchanger):
    #    def __init__(self, ranks, rank_in_group):
    #class HaloExchangerAllGather(HaloExchanger):
    #    def __init__(self, ranks, rank_in_group, comm):
    #class HaloExchangerSendRecv(HaloExchanger):
    #    def __init__(self, ranks, rank_in_group):
    #class HaloExchangerPeer(HaloExchanger):
    #    def __init__(self, ranks, rank_in_group, peer_pool, explicit_nhwc, numSM=1):

    #halex = HaloExchangerAllGather(ranks, rank_in_group)
    #halex = HaloExchangerSendRecv(ranks, rank_in_group)

    halex = HaloExchangerPeer(ranks, rank_in_group, peer_pool, explicit_nhwc, numSM=1)
    #print("halex.signals = %s" % (str(halex.signals)))
    # Make sure peer memory halo exchanger has finished initializing flags on all ranks before proceeding
    #torch.cuda.synchronize()
    #torch.distributed.barrier()

    bt2 = n_way_spatial(halex, gt_bottleneck, gt, explicit_nhwc, world_size, rank, fp32_reduce=True)
    compare(gt, bt2)


if __name__ == "__main__":
    main()
