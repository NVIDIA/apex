import os
import torch
from maskrcnn_benchmark.modeling.backbone.resnet import Bottleneck
from maskrcnn_benchmark.layers.nhwc import nhwc_to_nchw_transform, nchw_to_nhwc_transform
from maskrcnn_benchmark.layers.nhwc.batch_norm import FrozenBatchNorm2d_NHWC
from apex.contrib.bottleneck import Bottleneck as FastBottleneck
from apex.contrib.bottleneck import SpatialBottleneck


def single_module_test(ref, rank, world_size, numtype, device, shape, fast, spatial_group_size, in_channels, bottleneck_channels, out_channels, num_groups, stride_in_1x1, stride, dilation, norm_func, nhwc):
    # inputs + modules
    with torch.no_grad():
        input_shape = [1, in_channels] + list(shape)
        x = torch.randn(input_shape, dtype=numtype, device=device)
        if nhwc:
            x = nchw_to_nhwc_transform(x).contiguous()
        x.requires_grad = True
        print(x.shape, x.stride())

        #if spatial_group_size > 1:
        #    fast = False # hack so fast bottleneck can be run against distributed bottleneck
        #if spatial_group_size == 1:
        #    fast = False

        if fast:
            if spatial_group_size == 1:
                bottleneck = FastBottleneck(
                    in_channels=in_channels,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilation=dilation,
                    explicit_nhwc=nhwc,
                    use_cudnn=True)
            else:
                bottleneck = SpatialBottleneck(
                    in_channels=in_channels,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilation=dilation,
                    explicit_nhwc=nhwc,
                    use_cudnn=True,
                    spatial_group_size=spatial_group_size)
        else:
            bottleneck = Bottleneck(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation,
                norm_func,
                nhwc,
                spatial_group_size)
        bottleneck = bottleneck.to(dtype=numtype,device=device)
        weights = dict(bottleneck.named_parameters())

        if ref is not None:
            ref_x, _, ref_weights = ref
            Hs,H = x.shape[1], ref_x.shape[1]
            assert(Hs*spatial_group_size == H), "Hs not a multiple of H"
            ref_x = ref_x[:,rank*Hs:(rank+1)*Hs,:,:]
            x.copy_(ref_x)
            assert(len(weights) == len(ref_weights)), "Reference weights and weights don't match"
            for k in weights.keys():
                weights[k].copy_(ref_weights[k])

    # forward
    out = bottleneck(x)
    
    # gradient output
    with torch.no_grad():
        grad_out = torch.randn_like(out)
        if ref is not None:
            _, ref_grad_out, _ = ref
            Hs,H = grad_out.shape[1], ref_grad_out.shape[1]
            assert(Hs*spatial_group_size == H), "Hs not a multiple of H"
            ref_grad_out = ref_grad_out[:,rank*Hs:(rank+1)*Hs,:,:]
            grad_out.copy_(ref_grad_out)

    # backward
    out.backward(grad_out)

    with torch.no_grad():
        dgrad = x.grad.detach()
        
        wgrad = {}
        for n,p in bottleneck.named_parameters():
            wgrad[n] = p.grad.detach()

    if world_size > 1:
        if spatial_group_size == 1:
            # broadcast x, grad_out and weights from rank 0
            with torch.no_grad():
                torch.distributed.broadcast(x,0)
                torch.distributed.broadcast(grad_out,0)
                for k in weights.keys():
                    torch.distributed.broadcast(weights[k],0)
        else:
            # gather dgrad (x.grad), sum wgrad (weights) and out
            N,Hs,W,C = dgrad.shape
            H = Hs * spatial_group_size
            dgrad_gathered = torch.empty((N,H,W,C),dtype=dgrad.dtype,device=dgrad.device)
            dgrad_tensors = [dgrad_gathered[:,i*Hs:(i+1)*Hs,:,:] for i in range(spatial_group_size)]
            torch.distributed.all_gather(dgrad_tensors, dgrad)
            dgrad = dgrad_gathered
            N,Hs,W,C = list(out.shape)
            H = Hs * spatial_group_size
            out_gathered = torch.empty((N,H,W,C),dtype=dgrad.dtype,device=dgrad.device)
            out_tensors= [out_gathered[:,i*Hs:(i+1)*Hs,:,:] for i in range(spatial_group_size)]
            torch.distributed.all_gather(out_tensors, out)
            out = out_gathered
            for k in wgrad.keys():
                w = wgrad[k].to(dtype=torch.float64)
                torch.distributed.all_reduce(w)
                wgrad[k].copy_(w.to(dtype=wgrad[k].dtype))
                #torch.distributed.all_reduce(wgrad[k])

    return x, out, grad_out, weights, dgrad, wgrad


def module_tests(rank, world_size, numtype, device, fast, spatial_group_sizes, init_args):
    r = []
    for ia in init_args:
        shape = ia[0:4]
        args = ia[4:]
        rr = []
        ref = None
        for spatial_group_size in spatial_group_sizes:
            N,H,W,C = shape
            H = H//spatial_group_size
            x, out, grad_out, weights, dgrad, wgrad = single_module_test(ref, rank, world_size, numtype, device, [H,W], fast, spatial_group_size, *args)
            if ref is None:
                assert(spatial_group_size == 1), "Wrong reference weights"
                ref = x, grad_out, weights
            if rank == 0:
                rr.append( (out, dgrad, wgrad) )
            if world_size > 1: torch.distributed.barrier()
        r.append(rr)
    return r


def main():
    total_num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = total_num_gpus > 1
    ngpus = torch.cuda.device_count()

    if distributed:
        torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        is_master = True if rank == 0 else False
        local_rank = rank % ngpus
        torch.cuda.set_device(local_rank)
        spatial_group_size = total_num_gpus
    else:
        rank, local_rank, is_master, world_size, spatial_group_size = 0, 0, True, 1, 1

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    norm_func = FrozenBatchNorm2d_NHWC

    init_args = [
        (1, 200, 336, 64, 64, 64, 256, 1, True, 1, 1, norm_func, True),
        (1, 200, 336, 256, 256, 64, 256, 1, True, 1, 1, norm_func, True),
        (1, 200, 336, 256, 256, 128, 512, 1, True, 2, 1, norm_func, True),
        (1, 100, 168, 512, 512, 128, 512, 1, True, 1, 1, norm_func, True),
        (1, 100, 168, 512, 512, 256, 1024, 1, True, 2, 1, norm_func, True),
        (1, 50, 84, 1024, 1024, 256, 1024, 1, True, 1, 1, norm_func, True),
        (1, 50, 84, 1024, 1024, 512, 2048, 1, True, 2, 1, norm_func, True),
        (1, 25, 42, 2048, 2048, 512, 2048, 1, True, 1, 1, norm_func, True),
        (1, 336, 200, 64, 64, 64, 256, 1, True, 1, 1, norm_func, True),
        (1, 336, 200, 256, 256, 64, 256, 1, True, 1, 1, norm_func, True),
        (1, 336, 200, 256, 256, 128, 512, 1, True, 2, 1, norm_func, True),
        (1, 168, 100, 512, 512, 128, 512, 1, True, 1, 1, norm_func, True),
        (1, 168, 100, 512, 512, 256, 1024, 1, True, 2, 1, norm_func, True),
        (1, 84, 50, 1024, 1024, 256, 1024, 1, True, 1, 1, norm_func, True),
        (1, 84, 50, 1024, 1024, 512, 2048, 1, True, 2, 1, norm_func, True),
        (1, 42, 25, 2048, 2048, 512, 2048, 1, True, 1, 1, norm_func, True),
        ]
    init_args = init_args[0:1]

    # pad H to account for spatial distribution 
    padded_init_args = []
    for ia in init_args:
        N,H,W,C = ia[0:4]
        m = spatial_group_size * H // (25 if H < W else 42)
        H = ((H + m - 1) // m) * m
        args = tuple( [N,H,W,C] + list(ia[4:]) )
        padded_init_args.append(args)
    init_args = padded_init_args
    if rank == 0:
        for ia in init_args:
            print(ia)

    spatial_group_sizes = [1]
    if spatial_group_size > 1:
        spatial_group_sizes.append(spatial_group_size)

    numtype, device, fast = torch.float16, 'cuda', True
    r = module_tests(rank, world_size, numtype, device, fast, spatial_group_sizes, init_args)
    if world_size > 1: torch.distributed.barrier()
    if rank == 0:
        for rr in r:
            print("***")
            for out, dgrad, wgrad in rr:
                gr = [("out",out.norm(p=2,dtype=torch.float64).item())]
                gr = gr + [("dgrad",dgrad.norm(p=2,dtype=torch.float64).item())]
                gr = gr + [(k+".wgrad",wgrad[k].norm(p=2,dtype=torch.float64).item()) for k in wgrad.keys()]
                print(gr)
            if len(rr) == 2:
                out1, dgrad1, wgrad1 = rr[0]
                out2, dgrad2, wgrad2 = rr[1]

                rtol = 1e-1
                out_atol = out1.abs().max().item() * rtol
                dgrad_atol = dgrad1.abs().max().item() * rtol
                wgrad_atol = {}
                for k in wgrad1.keys():
                    wgrad_atol[k] = wgrad1[k].abs().max().item() * rtol

                gr = [("out",torch.allclose(out1,out2,rtol,out_atol,equal_nan=True))]
                gr = gr + [("dgrad",torch.allclose(dgrad1,dgrad2,rtol,dgrad_atol,equal_nan=True))]
                gr = gr + [(k+".wgrad",torch.allclose(wgrad1[k],wgrad2[k],rtol,wgrad_atol[k],equal_nan=True)) for k in wgrad1.keys()]
                print(gr)

                gr = [("out",(out1-out2).norm(p=2,dtype=torch.float64).item())]
                gr = gr + [("dgrad",(dgrad1-dgrad2).norm(p=2,dtype=torch.float64).item())]
                gr = gr + [(k+".wgrad",(wgrad1[k]-wgrad2[k]).norm(p=2,dtype=torch.float64).item()) for k in wgrad1.keys()]
                print(gr)

                N,H,W,C = out1.shape
                Hs = H // spatial_group_size
                Ht = Hs-2
                print("out1@%d:%d=%s" % (Ht,H,str(out1[0,Ht,:8,:5])))
                print("out2@%d:%d=%s" % (Ht,H,str(out2[0,Ht,:8,:5])))
                Ht = Hs-1
                print("out1@%d:%d=%s" % (Ht,H,str(out1[0,Ht,:8,:5])))
                print("out2@%d:%d=%s" % (Ht,H,str(out2[0,Ht,:8,:5])))
                Ht = Hs
                print("out1@%d:%d=%s" % (Ht,H,str(out1[0,Ht,:8,:5])))
                print("out2@%d:%d=%s" % (Ht,H,str(out2[0,Ht,:8,:5])))
                Ht = Hs+1
                print("out1@%d:%d=%s" % (Ht,H,str(out1[0,Ht,:8,:5])))
                print("out2@%d:%d=%s" % (Ht,H,str(out2[0,Ht,:8,:5])))

                N,H,W,C = dgrad1.shape
                Hs = H // spatial_group_size
                Ht = Hs-2
                print("dgrad1@%d:%d=%s" % (Ht,H,str(dgrad1[0,Ht,:8,:5])))
                print("dgrad2@%d:%d=%s" % (Ht,H,str(dgrad2[0,Ht,:8,:5])))
                Ht = Hs-1
                print("dgrad1@%d:%d=%s" % (Ht,H,str(dgrad1[0,Ht,:8,:5])))
                print("dgrad2@%d:%d=%s" % (Ht,H,str(dgrad2[0,Ht,:8,:5])))
                Ht = Hs
                print("dgrad1@%d:%d=%s" % (Ht,H,str(dgrad1[0,Ht,:8,:5])))
                print("dgrad2@%d:%d=%s" % (Ht,H,str(dgrad2[0,Ht,:8,:5])))
                Ht = Hs+1
                print("dgrad1@%d:%d=%s" % (Ht,H,str(dgrad1[0,Ht,:8,:5])))
                print("dgrad2@%d:%d=%s" % (Ht,H,str(dgrad2[0,Ht,:8,:5])))


    if world_size > 1: torch.distributed.barrier()


if __name__ == "__main__":
    main()
