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
  from amp_C import multi_tensor_axpby
  from apex.multi_tensor_apply import MultiTensorApply
  disabled = False
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestMultiTensorApply.  ImportError was ", err)
  disabled = True


class TestMultiTensorAxpby(unittest.TestCase):

    def setUp(self):
        common_init(self)

        self.a = 2.0
        self.b = 8.0
        self.xval = 4.0
        self.yval = 16.0
        self.overflow_buf = torch.cuda.IntTensor(1).zero_()
        self.ref = torch.cuda.FloatTensor([136.0])

    def tearDown(self):
        pass

    # The tensor creation here is written for convenience, not speed.
    def axpby(self, sizea, sizeb, applier, repeat_tensors,
              x_type, y_type, out_type, inplace=False):
        self.overflow_buf.zero_()
        t1 = torch.cuda.FloatTensor(sizea).fill_(1.0)
        t2 = torch.cuda.FloatTensor(sizeb).fill_(1.0)

        y_list = []
        for i in range(repeat_tensors):
            y_list += [t1.clone().to(y_type)*self.yval, t2.clone().to(y_type)*self.yval]

        x_list = [x.clone().to(x_type)*(self.xval/self.yval) for x in y_list]

        if inplace:
            out_list = y_list
        else:
            out_list = [out.clone().to(out_type)*3.0 for out in y_list]

        applier(multi_tensor_axpby, self.overflow_buf, [x_list, y_list, out_list], self.a, self.b, -1)

        self.assertTrue(all([torch.allclose(out, self.ref.to(out_type)) for out in out_list]),
                        msg="{} {} {} {} {} {} {}".format(sizea, sizeb, repeat_tensors,
                        x_type, y_type, out_type, inplace))
        self.assertTrue(self.overflow_buf.item() == 0,
                        msg="{} {} {} {} {} {} {}".format(sizea, sizeb, repeat_tensors,
                        x_type, y_type, out_type, inplace))

    # def find_inf(self, sizea, sizeb, applier, repeat_tensors, in_type, out_type, t, ind, val, inplace=False):
    #     self.overflow_buf.zero_()
    #     a = torch.cuda.FloatTensor(sizea).fill_(self.scale)
    #     b = torch.cuda.FloatTensor(sizeb).fill_(self.scale)

    #     out_list = []
    #     for i in range(repeat_tensors):
    #         out_list += [a.clone().to(out_type), b.clone().to(out_type)]

    #     if inplace:
    #         in_list = out_list
    #     else:
    #         in_list = [out.clone().to(in_type) for out in out_list]

    #     applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1./self.scale)

    #     self.overflow_buf.zero_()
    #     in_list[t][ind] = val
    #     applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1./self.scale)
    #     self.assertTrue(self.overflow_buf.item())

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
              for x_type in (torch.float32, torch.float16):
                for y_type in (torch.float32, torch.float16):
                  for out_type in (torch.float32, torch.float16):
                    for inplace in (True, False):
                      if inplace is True and (y_type is not out_type):
                        continue
                      else:
                        self.axpby(sizea, sizeb, applier, repeat,
                                   x_type, y_type, out_type, inplace=inplace)
                      # self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                      #               0, 0, float('nan'), inplace=inplace)
                      # self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                      #               2*repeat-1, sizeb-1, float('inf'), inplace=inplace)
                      # self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                      #              2*(repeat//2), sizea//2, float('inf'), inplace=inplace)



if __name__ == '__main__':
    unittest.main()
