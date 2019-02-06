import unittest

import torch
import torch.nn as nn

from apex.fp16_utils import FP16Model


class DummyBlock(nn.Module):
    def __init__(self):
        super(DummyBlock, self).__init__()

        self.conv = nn.Conv2d(10, 10, 2)
        self.bn = nn.BatchNorm2d(10, affine=True)

    def forward(self, x):
        return self.conv(self.bn(x))


class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 2)
        self.bn1 = nn.BatchNorm2d(10, affine=False)
        self.db1 = DummyBlock()
        self.db2 = DummyBlock()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.db1(out)
        out = self.db2(out)
        return out


class DummyNetWrapper(nn.Module):
    def __init__(self):
        super(DummyNetWrapper, self).__init__()

        self.bn = nn.BatchNorm2d(3, affine=True)
        self.dn = DummyNet()

    def forward(self, x):
        return self.dn(self.bn(x))


class TestFP16Model(unittest.TestCase):
    def setUp(self):
        self.N = 64
        self.C_in = 3
        self.H_in = 16
        self.W_in = 32
        self.in_tensor = torch.randn((self.N, self.C_in, self.H_in, self.W_in)).cuda()
        self.orig_model = DummyNetWrapper().cuda()
        self.fp16_model = FP16Model(self.orig_model)

    def test_params_and_buffers(self):
        exempted_modules = [
            self.fp16_model.network.bn,
            self.fp16_model.network.dn.db1.bn,
            self.fp16_model.network.dn.db2.bn,
        ]
        for m in self.fp16_model.modules():
            expected_dtype = torch.float if (m in exempted_modules) else torch.half
            for p in m.parameters(recurse=False):
                assert p.dtype == expected_dtype
            for b in m.buffers(recurse=False):
                assert b.dtype in (expected_dtype, torch.int64)

    def test_output_is_half(self):
        out_tensor = self.fp16_model(self.in_tensor)
        assert out_tensor.dtype == torch.half

