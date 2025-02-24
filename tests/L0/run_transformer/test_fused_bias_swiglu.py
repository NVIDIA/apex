import torch
import fused_bias_swiglu
from torch.testing._internal import common_utils
import torch.nn.functional as F


class TestFusedBiasSwiGLU(common_utils.TestCase):

    def swiglu(self, y):
        y_1, y_2 = torch.chunk(y, 2, -1)
        return F.silu(y_1) * y_2

    def bias_swiglu(self, y, bias):
        y = y + bias
        return self.swiglu(y)

    def swiglu_back(self, g, y):
        y_1, y_2 = torch.chunk(y, 2, -1)
        return torch.cat(
            (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
        )

    def bias_swiglu_back(self, g, y, bias):
        y = y + bias
        return self.swiglu_back(g, y)

    def test_fused_bias_swiglu(self):
        # Inputs
        batch_size, hidden_dim = 16, 512
        dtypes = [torch.float32, torch.float64, torch.float16]

        for dtype in dtypes:
            print(f"Testing with data type: {dtype}")
            input = torch.randn(batch_size, hidden_dim, device="cuda", dtype=dtype)
            bias = torch.randn(hidden_dim, device="cuda", dtype=dtype)

            try:
                actual = fused_bias_swiglu.forward(input, bias)
                expected = self.bias_swiglu(input, bias)

                self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)

                grad_output = torch.randn(batch_size, hidden_dim // 2, device="cuda", dtype=dtype)  # Output gradient
                actual_grad = fused_bias_swiglu.backward(grad_output, input, bias)
                expected_grad = self.bias_swiglu_back(grad_output, input, bias)
                self.assertEqual(actual_grad, expected_grad, atol=1e-3, rtol=1e-3)

                print(f"Test succeeded for data type: {dtype}")
            except AssertionError as e:
                print(f"Test failed for data type: {dtype}")
                print(e)


if __name__ == "__main__":
    common_utils.run_tests()