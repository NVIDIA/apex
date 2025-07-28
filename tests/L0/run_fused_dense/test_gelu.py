from apex import fused_dense 
import torch
import torch.nn.functional as F
import unittest


class FusedDenseGeluDenseTest(unittest.TestCase):

    def test_fused_dense_gelu_dense(self) :
        seed = 0
        torch.manual_seed(seed)
        batch_size   = 4
        in_features  = 3
        intermediate_features = 3
        out_features = 2

        #tst_dtype = torch.float8_e4m3
        # tst_dtype = torch.float8_e5m2
        tst_dtype = torch.float16

        I = torch.randn(batch_size, in_features, dtype=tst_dtype, device='cuda').requires_grad_(True)

        denseGelu = fused_dense.FusedDenseGeluDense(in_features, intermediate_features, out_features)
        denseGelu.to(dtype=tst_dtype)
        denseGelu.cuda()

        #get weight and bias from the denseGelu module
        W1 = denseGelu.weight1
        b1 = denseGelu.bias1
        W2 = denseGelu.weight2
        b2 = denseGelu.bias2

        y_tst = denseGelu(I.clone().detach().requires_grad_(True))

        C1  = torch.matmul(I, W1.t())+b1
        gelu_output = F.gelu(C1)
        y_ref = torch.matmul(gelu_output, W2.t())+b2
        torch.testing.assert_close(y_ref,  y_tst,  atol=1e-3, rtol=1e-3, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
