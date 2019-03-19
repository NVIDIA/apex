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
  from amp_C import multi_tensor_scale 
  from apex.multi_tensor_apply import MultiTensorApply
  disabled = False
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestMultiTensorApply.  ImportError was ", err)
  disabled = True


class TestMultiTensorScale(unittest.TestCase):

    def setUp(self):
        common_init(self)
        self.scale = 4.0
        self.overflow_buf = torch.cuda.IntTensor(1).zero_()
        self.ref = torch.cuda.FloatTensor([1.0])

    def tearDown(self):
        pass

    # The tensor creation here is written for convenience, not speed.
    def downscale(self, sizea, sizeb, applier, repeat_tensors, in_type, out_type, inplace=False):
        self.overflow_buf.zero_()
        a = torch.cuda.FloatTensor(sizea).fill_(self.scale)
        b = torch.cuda.FloatTensor(sizeb).fill_(self.scale)

        out_list = []
        for i in range(repeat_tensors):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1./self.scale)

        self.assertTrue(all([torch.allclose(out, self.ref.to(out_type)) for out in out_list]))
        self.assertTrue(self.overflow_buf.item() == 0)
 
    def find_inf(self, sizea, sizeb, applier, repeat_tensors, in_type, out_type, t, ind, val, inplace=False):
        self.overflow_buf.zero_()
        a = torch.cuda.FloatTensor(sizea).fill_(self.scale)
        b = torch.cuda.FloatTensor(sizeb).fill_(self.scale)

        out_list = []
        for i in range(repeat_tensors):
            out_list += [a.clone().to(out_type), b.clone().to(out_type)]

        if inplace:
            in_list = out_list
        else:
            in_list = [out.clone().to(in_type) for out in out_list]

        applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1./self.scale)

        self.overflow_buf.zero_()
        in_list[t][ind] = val
        applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1./self.scale)
        self.assertTrue(self.overflow_buf.item())

    # Currently, the fused kernel gives a hard error if you attempt to downscale
    # into fp16 output, which imo is the desired behavior.  Maybe someday we
    # will learn otherwise.
    # @unittest.skipIf(disabled, "amp_C is unavailable")
    # def test_fp16_to_fp16(self):
    #     self.downscale(self.fp16, self.fp16, self.fp16_ref)
    # 
    # @unittest.skipIf(disabled, "amp_C is unavailable")
    # def test_fp32_to_fp16(self):
    #     self.downscale(self.fp32, self.fp16, self.fp16_ref)

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
                for out_type in (torch.float32, torch.float16):
                  for inplace in (True, False):
                    if inplace is True and (out_type is not in_type):
                      continue
                    else:
                      self.downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=inplace)
                      self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                                    0, 0, float('nan'), inplace=inplace)
                      self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                                    2*repeat-1, sizeb-1, float('inf'), inplace=inplace)
                      self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                                   2*(repeat//2), sizea//2, float('inf'), inplace=inplace)



if __name__ == '__main__':
    unittest.main()
