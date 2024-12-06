from apex import FusedDenseGeluDense
import torch
import torch.nn.functional as F

batch_size   = 4
in_features  = 3
intermediate_features = 3
out_features = 2

#tst_dtype = torch.float8_e4m3
# tst_dtype = torch.float8_e5m2
tst_dtype = torch.float16

# I = torch.randn(batch_size, in_features, dtype=tst_dtype, device='cuda')
I = torch.tensor([[1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.]],dtype=tst_dtype, device='cuda')

# W = torch.randn(out_features, in_features, dtype=tst_dtype, device='cuda')
W = torch.tensor([[1., 1. , 1. , 1. ],
                  [2., 2. , 2. , 2. ],
                  [3., 3. , 3. , 3. ]],dtype=tst_dtype, device='cuda')

# b = torch.randn(in_features, dtype=tst_dtype, device='cuda')
b = torch.tensor([1, 1, 1], dtype=tst_dtype, device='cuda')

print("Torch-A:\n", I)
print("Torch-B:\n", W)
print("Torch-b:\n", b)

C  = torch.matmul(I, W.t())+b
gelu_output = F.gelu(C)
print("Torch-C:\n", C)
print("Torch-Geli:\n", gelu_output)

denseGlue = FusedDenseGeluDense.fused_dense_gelu_dense_function(in_features, intermediate_features, out_features)
denseGlue.to(dtype=tst_dtype)
denseGlue.cuda()
y_tst = denseGlue(I)

print("Torch-aC:\n", aC)
print("GELU tensor:\n", gelu_output)


