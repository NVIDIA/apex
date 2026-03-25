import unittest
from functools import wraps

import torch

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    HAS_APEX = False
else:
    HAS_APEX = True

from torch.testing._internal.common_device_type import largeTensorTest

INT32_MAX = 2_147_483_647
LARGE_NUMEL = INT32_MAX + 1

@unittest.skipIf(not HAS_APEX, "`apex` is not found.")
class LargeTensorL2NormTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.noop_flag = torch.zeros([1], dtype=torch.int32, device="cuda")

    def _make_large_tensor(self, dtype=torch.float16):
        tensor = torch.zeros(LARGE_NUMEL, dtype=dtype, device="cuda")
        tensor[0] = 3
        tensor[-1] = 4
        return tensor

    @largeTensorTest("5GB", "cuda")
    def test_multi_tensor_l2norm_large_tensor(self):
        tensor = self._make_large_tensor(torch.float16)

        expected = torch.norm(tensor, 2.0).float().unsqueeze(0)
        actual, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm,
            self.noop_flag,
            [[tensor]],
            False,
        )

        torch.testing.assert_close(actual, expected)

    @largeTensorTest("5GB", "cuda")
    def test_multi_tensor_l2norm_mp_large_tensor(self):
        tensor = self._make_large_tensor(torch.float16)

        expected = torch.norm(tensor, 2.0).float().unsqueeze(0)
        actual, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm_mp,
            self.noop_flag,
            [[tensor]],
            False,
        )

        torch.testing.assert_close(actual, expected)

    @largeTensorTest("9GB", "cuda")
    def test_multi_tensor_l2norm_scale_large_tensor(self):
        tensor = self._make_large_tensor(torch.float16)
        scaled = torch.empty_like(tensor)
        scale = 0.5

        expected = torch.norm(tensor, 2.0).float().unsqueeze(0)
        actual, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm_scale,
            self.noop_flag,
            [[tensor], [scaled]],
            scale,
            False,
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(scaled[0].item(), 1.5)
        self.assertEqual(scaled[-1].item(), 2.0)


if __name__ == "__main__":
    unittest.main()