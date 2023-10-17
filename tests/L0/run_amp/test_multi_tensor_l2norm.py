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
  from amp_C import multi_tensor_l2norm
  from apex.multi_tensor_apply import MultiTensorApply
  disabled = False
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestMultiTensorApply.  ImportError was ", err)
  disabled = True


class TestMultiTensorL2Norm(unittest.TestCase):

    def setUp(self):
        common_init(self)
        self.val = 4.0
        self.overflow_buf = torch.cuda.IntTensor(1).zero_()

    def tearDown(self):
        pass

    # The tensor creation here is written for convenience, not speed.
    def l2norm(self, sizea, sizeb, applier, repeat_tensors, in_type, per_tensor):
        self.overflow_buf.zero_()
        a = torch.cuda.FloatTensor(sizea).fill_(self.val)
        b = torch.cuda.FloatTensor(sizeb).fill_(self.val)

        in_list = []
        for i in range(repeat_tensors):
            in_list += [a.clone().to(in_type), b.clone().to(in_type)]

        if per_tensor:
            norm, norm_per_tensor = applier(multi_tensor_l2norm, self.overflow_buf, [in_list], True)
            normab = torch.cat((a.norm().view(1), b.norm().view(1)))
            norm_per_tensor = norm_per_tensor.view(-1, 2)
        else:
            norm, _ = applier(multi_tensor_l2norm, self.overflow_buf, [in_list], True)

        reference = torch.cuda.FloatTensor((sizea + sizeb)*repeat_tensors).fill_(self.val).norm()

        torch.testing.assert_close(norm, reference.broadcast_to(norm.shape))
        if per_tensor:
          torch.testing.assert_close(norm_per_tensor, normab.broadcast_to(norm_per_tensor.shape))
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
                  self.l2norm(sizea, sizeb, applier, repeat, in_type, per_tensor)



if __name__ == '__main__':
    unittest.main()
