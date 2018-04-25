import torch
import sys
import apex._C
import numpy as np
from compare import compare
from norm import pt_norm, get_norm_shape


torch.manual_seed(2)
torch.cuda.manual_seed(2)
# torch.cuda.manual_seed_all(2)
torch.set_printoptions(precision=10)

sizes = [
    # (3,  512, 1024),
    # (3,  512, 1536),
    # (3,  768, 1536),
    # (3,  768, 2048),
    # (3, 1024, 2048),
    # (1, 1024, 4096),
    # (1, 2048, 8192),
    # (1, 4096, 4096), # this is not one of the fairseq sizes, just a reference benchmark.
    (4096, 4096, 1), # this is not one the fairseq sizes, just a reference benchmark.
    # (353, 55, 353), # this is not one of the fairseq sizes, just a reference benchmark.
    ]

# rows = 3
# cols = 512
# fast = 1024
HALF = True
RAND = True
dim = 0
    

for rows, cols, fast in sizes:
    dims = rows, cols, fast
     
    print("\n\nTESTING dims = {}\n\n".format(dims))

    if RAND:
        pt_in = 1.*torch.cuda.FloatTensor(*dims).uniform_()
        g = torch.cuda.FloatTensor(*get_norm_shape(pt_in, dim)).uniform_()
    else:
        pt_in = torch.cuda.FloatTensor(*dims).fill_(1.)
        g = torch.cuda.FloatTensor(*get_norm_shape(pt_in, dim)).fill_(6.0)
    
    # per_col = torch.arange(1,cols+1).cuda()
    # print((rows*per_col*per_col).sqrt())
    # pt_in *= per_col
    
    cuda_out   =   torch.cuda.FloatTensor(*dims).fill_(0.)
    cuda_norms =   torch.cuda.FloatTensor(*get_norm_shape(pt_in, dim)).fill_(0.)
    
    # Save a copy of the input as float
    pt_in_fp32 = pt_in.clone()
    g_fp32     = g.clone()
    
    if HALF:
        pt_in    =    pt_in.half()
        g        =        g.half()
        cuda_out = cuda_out.half()
    
    apex._C.weight_norm_fwd(cuda_out, cuda_norms, pt_in, g, dim)
    torch.cuda.synchronize()
    # quit()

    print("type(cuda_out) = {}\n".format(type(cuda_out)))
    
    rownorms      = pt_norm(pt_in, dim)
    rownorms_fp32 = pt_norm(pt_in_fp32, dim)
    
    print("rownorms_fp32:")
    print(rownorms_fp32)
    print("cuda_norms"    )
    print(cuda_norms   )
    
    # rownorms is broadcast; torch.div(pt_in, rownorms) and pt_in/rownorms work the same way
    pt_out         = pt_in*(g/rownorms)
    pt_out_control = pt_in_fp32*(g_fp32/rownorms_fp32)
    
    compare(cuda_out, pt_out, pt_out_control, rows)
