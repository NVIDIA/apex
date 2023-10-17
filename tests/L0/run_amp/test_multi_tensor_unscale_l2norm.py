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
  from amp_C import multi_tensor_unscale_l2norm
  from apex.multi_tensor_apply import MultiTensorApply
  disabled = False
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestMultiTensorApply.  ImportError was ", err)
  disabled = True


class TestMultiTensorUnscaleL2Norm(unittest.TestCase):

    def setUp(self):
        common_init(self)
        self.val = 4.0
        self.inv_scale = 0.5
        self.inv_scale_cuda = torch.tensor([self.inv_scale], dtype=torch.float32, device='cuda')
        self.overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')

    def tearDown(self):
        pass

    # The tensor creation here is written for convenience, not speed.
    def unscale_l2norm(self, sizea, sizeb, applier, repeat_tensors, in_type, per_tensor):
        self.overflow_buf.zero_()
        a = torch.full([sizea], self.val, dtype=torch.float32, device='cuda')
        b = torch.full([sizeb], self.val, dtype=torch.float32, device='cuda')

        in_list = []
        for i in range(repeat_tensors):
            in_list += [a.clone().to(in_type), b.clone().to(in_type)]

        if per_tensor:
            norm, norm_per_tensor = applier(multi_tensor_unscale_l2norm, self.overflow_buf, [in_list], self.inv_scale_cuda, True)
            normab = torch.cat(((a * self.inv_scale).norm().view(1), (b * self.inv_scale).norm().view(1)))
            norm_per_tensor = norm_per_tensor.view(-1, 2)
        else:
            norm, _ = applier(multi_tensor_unscale_l2norm, self.overflow_buf, [in_list], self.inv_scale_cuda, True)

        reference = torch.full([(sizea + sizeb)*repeat_tensors], self.val * self.inv_scale, dtype=torch.float32, device='cuda').norm()

        torch.testing.assert_close(norm, reference)
        if per_tensor:
            torch.testing.assert_close(norm_per_tensor, normab)
        self.assertTrue(self.overflow_buf.item() == 0)

    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_fuzz(self):
        input_size_pairs = (
            (7777*77, 555*555),
            (777, 555),
            (555, 2048*32+1),
            (2048*32+1, 555),
            (555, 2048*32),
            (2048*32, 555),
            (33333, 555),
            (555, 33333))
        appliers = (
            MultiTensorApply(2048*32),
            MultiTensorApply(333),
            MultiTensorApply(33333))
        repeat_tensors = (
            1,
            55)

        for sizea, sizeb in input_size_pairs:
          for applier in appliers:
            for repeat in repeat_tensors:
              for in_type in (torch.float32, torch.float16):
                for per_tensor in (False, True):
                  self.unscale_l2norm(sizea, sizeb, applier, repeat, in_type, per_tensor)



if __name__ == '__main__':
    unittest.main()
