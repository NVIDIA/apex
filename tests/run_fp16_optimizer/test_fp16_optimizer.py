import unittest

import functools as ft
import itertools as it

import torch
from apex.fp16_utils import FP16_Optimizer

# Currently no-ops (tested via examples).
# FP16_Optimizer to be deprecated and moved under unified Amp API.
class TestFP16Optimizer(unittest.TestCase):
    def setUp(self):
        N, D_in, D_out = 64, 1024, 16
        self.N = N
        self.D_in = D_in
        self.D_out = D_out
        self.x = torch.randn((N, D_in), dtype=torch.float16, device='cuda')
        self.y = torch.randn((N, D_out), dtype=torch.float16, device='cuda')
        self.model = torch.nn.Linear(D_in, D_out).cuda().half()

    # def tearDown(self):
    #     pass

    def test_minimal(self):
        pass

    def test_minimal_static(self):
        pass

    def test_minimal_dynamic(self):
        pass

    def test_closure(self):
        pass

    def test_closure_dynamic(self):
        pass

    def test_save_load(self):
        pass

if __name__ == '__main__':
    unittest.main()
