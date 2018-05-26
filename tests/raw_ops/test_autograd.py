import torch
from torch.autograd import Variable
from apex.fp16_utils import Fused_Weight_Norm
from compare import compare
from norm import pt_norm, get_norm_shape

torch.manual_seed(2)
torch.cuda.manual_seed(2)
# torch.cuda.manual_seed_all(2)
torch.set_printoptions(precision=10)

rows = 321 # 1    
cols = 33  # 4096 
fast = 185 # 4096 
dims = rows, cols, fast

dim = 0
CUDA_HALF = True
RAND      = True # If false, input gradients (the result of the backward pass) 
                 # should be analytically zero.

# Loss will be computed via (output*elementwise).sum().
# This means that output gradients in the backward pass will be equal
# to elementwise, so by manipulating elementwise, we have easy 
# fine-grained control over the output gradients we'd like to use for
# testing purposes.
# 
# The alternative is just to create the output_gradients manually 
# and call output.backward(gradient=output_gradients), 
# as is done in test_backward.py.
# But I wanted a minimal working sample similar to an "actual" use case, 
# where gradients are computed by calling backward() on a scalar Loss.

if RAND:
    # With std=6.0, I observe the pytorch fp16 ops going unstable (sometimes)
    # while the fused kernel remains stable.
    pt_in_fp32       = torch.cuda.FloatTensor(*dims      ).normal_(std=1.0)
    norm_shape = get_norm_shape(pt_in_fp32, dim)
    pt_g_fp32        = torch.cuda.FloatTensor(*norm_shape).normal_(std=1.0)
    elementwise_fp32 = torch.cuda.FloatTensor(*dims      ).normal_(std=1.0)
else:
    pt_in_fp32       = torch.cuda.FloatTensor(*dims      ).fill_(1.0)
    norm_shape = get_norm_shape(pt_in_fp32, dim)
    pt_g_fp32        = torch.cuda.FloatTensor(*norm_shape).fill_(2.0)
    elementwise_fp32 = torch.cuda.FloatTensor(*dims      ).fill_(0.5)

pt_in_fp16       = pt_in_fp32.half()
cd_in_prec       = pt_in_fp32.clone()
pt_g_fp16        = pt_g_fp32.half()
cd_g_prec        = pt_g_fp32.clone()
elementwise_fp16 = elementwise_fp32.half()
elementwise_prec = elementwise_fp32.clone()

if CUDA_HALF:
    cd_in_prec       = cd_in_prec.half()
    cd_g_prec        = cd_g_prec.half()
    elementwise_prec = elementwise_prec.half()

pt_in_fp32 = Variable(pt_in_fp32 , requires_grad=True)
pt_in_fp16 = Variable(pt_in_fp16 , requires_grad=True)
cd_in_prec = Variable(cd_in_prec , requires_grad=True)

pt_g_fp32 = Variable(pt_g_fp32 , requires_grad=True)
pt_g_fp16 = Variable(pt_g_fp16 , requires_grad=True)
cd_g_prec = Variable(cd_g_prec , requires_grad=True)

elementwise_fp32 = Variable(elementwise_fp32, requires_grad=False)
elementwise_fp16 = Variable(elementwise_fp16, requires_grad=False)
elementwise_prec = Variable(elementwise_prec, requires_grad=False)

torch.cuda.nvtx.range_push("fp16 forward, {}".format(pt_in_fp16.size()))
pt_norms_fp16 = pt_norm(pt_in_fp16, dim)
pt_out_fp16 = pt_in_fp16*(pt_g_fp16/pt_norms_fp16) 
torch.cuda.nvtx.range_pop()
# torch.cuda.synchronize()

torch.cuda.nvtx.range_push("fp32 forward, {}".format(pt_in_fp32.size()))
pt_norms_fp32 = pt_norm(pt_in_fp32, dim)
pt_out_fp32 = pt_in_fp32*(pt_g_fp32/pt_norms_fp32)
torch.cuda.nvtx.range_pop()
# torch.cuda.synchronize()

# print("pt_norms_fp16    = ", pt_norms_fp16   )
# print("pt_norms_fp32 = ", pt_norms_fp32)

# print( "cd_in_prec.data_ptr = {:x}".format(cd_in_prec.data_ptr()))

# print("elementwise_fp16 = ", elementwise_fp16)

cd_in_contig = cd_in_prec.contiguous()
# Deliberately make noncontig to see if fused_norm
# will handle the error
# cd_in_contig = cd_in_contig[:,0:5]
# print(type(cd_in_contig))
torch.cuda.nvtx.range_push("kernel forward")
fused_weight_norm = Fused_Weight_Norm.apply
cd_out_prec = fused_weight_norm(cd_in_contig, cd_g_prec, dim)
torch.cuda.nvtx.range_pop()
# torch.cuda.synchronize()

# print("type(cd_out_prec.data) = ", type(cd_out_prec.data))
# print("cd_out_prec.data_ptr = {:x}".format(cd_out_prec.data_ptr()))

print("\n\n\nCOMPARING FORWARD PASS RESULTS\n\n\n")
compare(cd_out_prec.data, 
        pt_out_fp16.data,
        pt_out_fp32.data,
        rows)

# It's ok to use elementwise_fp16 as a leaf in both the cuda and pytorch graphs.
# This sharing should not affect the computed gradients wrt pt_in_fp16 and cd_in_prec.
# However, just remember:  
# If we set requires_grad=True for elementwise_fp16, elementwise_fp16.grad.data
# will accumulate gradients during the backward passes for both the cd and pytorch Losses.
#
# I do need    v these parentheses          v             
Loss_cd_prec = (cd_out_prec*elementwise_prec).sum()
# print(L_cd_fp16)
Loss_pt_fp16 = (pt_out_fp16*elementwise_fp16).sum()
# print(L_pt_fp16)
Loss_pt_fp32 = (pt_out_fp32*elementwise_fp32).sum()
# print(L_pt_fp32)

torch.cuda.nvtx.range_push("kernel backward")
Loss_cd_prec.backward()
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("fp16 backward")
Loss_pt_fp16.backward()
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("fp32 backward")
Loss_pt_fp32.backward()
torch.cuda.nvtx.range_pop()

print("\n\n\nCOMPARING v GRADIENT RESULTS\n\n\n")
compare(cd_in_prec.grad.data, 
        pt_in_fp16.grad.data, 
        pt_in_fp32.grad.data, 
        rows)

print("\n\n\nCOMPARING g GRADIENT RESULTS\n\n\n")
compare(cd_g_prec.grad.data, 
        pt_g_fp16.grad.data, 
        pt_g_fp32.grad.data, 
        cd_g_prec.size(0))


