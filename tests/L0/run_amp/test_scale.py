import unittest

import functools as ft
import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT

try:
  import amp_C
  scale_check_overflow = amp_C.scale_check_overflow
  disabled = False
except ImportError as err:
  print("amp_C fused kernel unavailable, disabling TestScale.  ImportError was ", err)
  disabled = True


class TestScale(unittest.TestCase):

    def setUp(self):
        self.scale = 128.0
        self.nx = 999
        self.ny = 888
  
        self.overflow_buf = torch.cuda.IntTensor([0])
        self.fp16 = torch.ones((self.ny, self.nx), device='cuda', dtype=torch.float16)
        self.fp32 = torch.ones((self.ny, self.nx), device='cuda', dtype=torch.float32)
        self.fp16_ref = torch.ones((1, 1), device='cuda', dtype=torch.float16)
        self.fp32_ref = torch.ones((1, 1), device='cuda', dtype=torch.float32)

        common_init(self)

    def tearDown(self):
        pass

    def downscale_test(self, input, output, ref):
        self.overflow_buf.zero_()
        input.fill_(1.0)
        if input is not output:
            output.fill_(3.0)
        input.mul_(self.scale)
        scale_check_overflow(input, 1./self.scale, self.overflow_buf, output)
        self.assertTrue(torch.allclose(output, ref))
        self.assertTrue(self.overflow_buf.item() == 0)
 
    def find_inf_test(self, input, output, ref, x, y, val):
        self.overflow_buf.zero_()
        input.fill_(1.0)
        if input is not output:
            output.fill_(3.0)
        input[x,y] = val
        scale_check_overflow(input, 1./self.scale, self.overflow_buf, output)
        self.assertTrue(self.overflow_buf.item())

    # Currently, the fused kernel gives a hard error if you attempt to downscale
    # into fp16 output, which imo is the desired behavior.  Maybe someday we
    # will learn otherwise.
    # @unittest.skipIf(disabled, "amp_C is unavailable")
    # def test_fp16_to_fp16(self):
    #     self.downscale_test(self.fp16, self.fp16, self.fp16_ref)

    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_fp16_to_fp32(self):
        self.downscale_test(self.fp16, self.fp32, self.fp32_ref)

    # @unittest.skipIf(disabled, "amp_C is unavailable")
    # def test_fp32_to_fp16(self):
    #     self.downscale_test(self.fp32, self.fp16, self.fp16_ref)

    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_fp32_to_fp32(self):
        self.downscale_test(self.fp32, self.fp32, self.fp32_ref)

    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_fp16_to_fp32_find_inf_nan(self):
        self.find_inf_test(self.fp16, self.fp32, self.fp32_ref, 0, 0, float('nan'))
        self.find_inf_test(self.fp16, self.fp32, self.fp32_ref, self.ny//2, self.nx//2, float('inf'))
        self.find_inf_test(self.fp16, self.fp32, self.fp32_ref, self.ny-1, self.nx-1, float('nan'))

    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_fp32_to_fp32_find_inf_nan(self):
        self.find_inf_test(self.fp32, self.fp32, self.fp32_ref, 0, 0, float('inf'))
        self.find_inf_test(self.fp32, self.fp32, self.fp32_ref, self.ny//2, self.nx//2, float('nan'))
        self.find_inf_test(self.fp32, self.fp32, self.fp32_ref, self.ny-1, self.nx-1, float('inf'))


if __name__ == '__main__':
    unittest.main()
