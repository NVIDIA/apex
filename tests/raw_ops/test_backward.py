import torch
from torch.autograd import Variable
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
    (3,  768, 1536),
    # (3,  768, 2048),
    # (3, 1024, 2048),
    # (1, 1024, 4096),
    # (1, 2048, 8192),
    # (1, 4096, 4096), # this is not one of the fairseq sizes, just a reference benchmark.
    # (4096, 4096, 1), # this is not one of the fairseq sizes, just a reference benchmark.
    ]

# rows = 3
# cols = 512
# fast = 1024
HALF = True
RAND = True
dim = 2

for rows, cols, fast in sizes:
    dims = rows, cols, fast
    # Incoming gradient vectors we will use later
    # Need to create the fp16 versions as a half() copy of a Tensor first rather than
    # a Variable, because if you create pt_input_control as a Variable then say
    # pt_input_fp16 = pt_input_control.half(), you are accidentally making pt_input_fp16 part of 
    # pLpOutput_control's computational graph, instead of the leaf of its own separate graph.
    
    # Careful: if you initialize with torch.ones, the gradient wrt input becomes analytically zero.
    if RAND:
        pLpOutput_control = torch.cuda.FloatTensor(*dims      ).uniform_()*1.0 
        norm_shape = get_norm_shape(pLpOutput_control, dim)
        pLpg_control      = torch.cuda.FloatTensor(*norm_shape).uniform_()
        pt_input_control  = torch.cuda.FloatTensor(*dims      ).uniform_()
        pt_g_control      = torch.cuda.FloatTensor(*norm_shape).uniform_()
    else:
        pLpOutput_control = torch.cuda.FloatTensor(*dims      ).fill_(1.)
        norm_shape = get_norm_shape(pLpOutput_control, dim)
        pLpg_control      = torch.cuda.FloatTensor(*norm_shape).fill_(2.)
        pt_input_control  = torch.cuda.FloatTensor(*dims      ).fill_(4.0)
        pt_g_control      = torch.cuda.FloatTensor(*norm_shape).fill_(3.0)
    
    pLpOutput_fp16 = pLpOutput_control.clone()
    pLpg_fp16      = pLpg_control     .clone()
    pt_input_fp16  = pt_input_control .clone()
    pt_g_fp16      = pt_g_control     .clone()
    
    if HALF:
        pLpOutput_fp16 = pLpOutput_fp16.half()
        pLpg_fp16      = pLpg_fp16     .half() 
        pt_input_fp16  = pt_input_fp16 .half()
        pt_g_fp16      = pt_g_fp16     .half()
    
    pLpOutput_control = Variable(pLpOutput_control)
    pLpg_control      = Variable(pLpg_control     )
    pLpOutput_fp16    = Variable(pLpOutput_fp16   )
    pLpg_fp16         = Variable(pLpg_fp16        )
    
    pt_input_control = Variable(pt_input_control, requires_grad=True)
    pt_g_control     = Variable(pt_g_control    , requires_grad=True)
    pt_input_fp16    = Variable(pt_input_fp16   , requires_grad=True)
    pt_g_fp16        = Variable(pt_g_fp16       , requires_grad=True)
    
    # Do forward pass in fp16 and fp32
    pt_norms_fp16 = pt_norm(pt_input_fp16, dim)
    pt_norms_control = pt_norm(pt_input_control, dim)
    
    pt_output_fp16    = pt_input_fp16   *(pt_g_fp16   /pt_norms_fp16   )
    pt_output_control = pt_input_control*(pt_g_control/pt_norms_control)
    
    # Run the Cuda version
    pLpInput_cuda = torch.cuda.FloatTensor(*dims      ).fill_(0.)
    pLpg_cuda     = torch.cuda.FloatTensor(*norm_shape).fill_(0.)
    
    if HALF:
        pLpInput_cuda = pLpInput_cuda.half()
        pLpg_cuda     = pLpg_cuda    .half()
   
    torch.cuda.nvtx.range_push("kernel weight norm backward")
    apex._C.weight_norm_bwd(pLpInput_cuda, 
                            pLpg_cuda, 
                            pLpOutput_fp16,
                            pt_input_fp16, 
                            pt_g_fp16,
                            pt_norms_control.data,
                            dim)
    torch.cuda.nvtx.range_pop()
    
    print("grad_output:  ", pLpOutput_fp16.data)
    print(" grad_input:  ", pLpInput_cuda)
    print(" savedInput:  ", pt_input_fp16.data)
    print("pt_norms_control:  ", pt_norms_control.data)
    print("pt_norms_fp16:  ", pt_norms_fp16.data)
   
    torch.cuda.nvtx.range_push("pytorch fp16 backward")
    pt_output_fp16   .backward(gradient=pLpOutput_fp16   , create_graph=True)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("pytorch fp32 backward")
    pt_output_control.backward(gradient=pLpOutput_control, create_graph=True)
    torch.cuda.nvtx.range_pop()
    
    # pt_output_fp16 and pt_output_control are still saved, but
    # pt_output_fp16.grad and pt_output_control.grad are None at this point 
    # because the graph is freed in the backwards pass.  
    # Specifying create_/retain_ graph don't seem to force saving of 
    # either the intermediate variables or their gradients.
    
    print("Comparing gradients wrt v")
    torch.cuda.nvtx.range_push("compare pLpv")
    compare(pLpInput_cuda, pt_input_fp16.grad.data, pt_input_control.grad.data, rows)
    torch.cuda.nvtx.range_pop()
    
    print("Comparing gradients wrt g")
    torch.cuda.nvtx.range_push("compare pLpg")
    compare(pLpg_cuda, pt_g_fp16.grad.data, pt_g_control.grad.data, pLpg_cuda.size(0))
    torch.cuda.nvtx.range_pop()
    
    
