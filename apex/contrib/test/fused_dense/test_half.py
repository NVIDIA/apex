from apex import fused_dense
import torch

batch_size   = 5
in_features  = 4
out_features = 3

tst_dtype = torch.float8_e5m2

I = torch.randn(batch_size, in_features, dtype=tst_dtype, device='cuda')

W = torch.randn(in_features, out_features, dtype=tst_dtype, device='cuda')

b = torch.randn(out_features, dtype=tst_dtype, device='cuda')

print("Torch-A:\n", I)
print("Torch-B:\n", W)
print("Torch-b:\n", b)


aC = fused_dense.fused_dense_function(I, W, b)
print("Torch-aC:\n", aC)
torch.testing.assert_close(C,  aC,  atol=1e-3, rtol=1e-3, equal_nan=True)
