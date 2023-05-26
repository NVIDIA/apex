import unittest
import os

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests

SKIP_TEST = None
try:
    from apex import fused_dense
except ImportError as e:
    SKIP_TEST = e


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class FusedDenseTest(common_utils.TestCase):

    def _test_fused_dense(self, dtype, seed=0):

        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        torch.manual_seed(seed)

        seq_length = 512
        sequences = 3
        hidden_dim = 1024

        ref_inputs = torch.randn(sequences*seq_length, hidden_dim,
                                 dtype=dtype, device=torch.device("cuda")).requires_grad_(True)

        tst_inputs = ref_inputs.clone().detach().requires_grad_(True)
        dense = fused_dense.FusedDense(1024, 3072)
        dense.to(dtype=dtype)
        dense.cuda()

        y_tst = dense(tst_inputs)
        y_ref = torch.matmul(ref_inputs, dense.weight.t())+dense.bias
        dy = torch.randn_like(y_tst).to(dtype=dtype)
        y_tst.backward(dy)
        dw_ref = torch.matmul(dy.t(), ref_inputs)
        dx_ref = torch.matmul(dy, dense.weight.clone())
        db_ref = dy.sum(0, False)

        torch.testing.assert_close(
            ref_inputs,  tst_inputs,  atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            y_ref,  y_tst,  atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(
            dw_ref, dense.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(
            dx_ref, tst_inputs.grad, atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(
            db_ref, dense.bias.grad, atol=1e-3, rtol=1e-3, equal_nan=True)

    @common_utils.parametrize("dtype", [torch.half, torch.float, torch.bfloat16])
    def test_fused_dense(self, dtype):
        self._test_fused_dense(dtype)


instantiate_device_type_tests(FusedDenseTest, globals(), only_for=("cuda",))

if __name__ == "__main__":
    common_utils.run_tests()
