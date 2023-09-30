import unittest
import random
import math

import torch

try:
  import amp_C
  from amp_C import update_scale_hysteresis
  disabled = False
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestUpdateScaleHysteresis.  ImportError was ", err)
  disabled = True

def isfinite(val):
    return ((val >= torch.finfo(torch.float32).smallest_normal) and (val <= torch.finfo(torch.float32).max))

class TestUpdateScaleHysteresis(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def update_scale_hysteresis_body(self, init_scale, growth_factor, backoff_factor,
            growth_interval, hysteresis):
        scale_ref = float(init_scale)
        grow_tracker_ref = 0
        hysteresis_tracker_ref = 0

        scale = torch.tensor([init_scale], dtype=torch.float32, device='cuda')
        growth_tracker = torch.tensor([0], dtype=torch.int32, device='cuda')
        hysteresis_tracker = torch.tensor([hysteresis], dtype=torch.int32, device='cuda')

        # Infs appear for hysteresis-1 iterations, scale shouldn't change
        found_inf = torch.tensor([1], dtype=torch.float32, device='cuda')
        for i in range(hysteresis-1):
            update_scale_hysteresis(scale, growth_tracker, hysteresis_tracker,
                    found_inf, growth_factor, backoff_factor, growth_interval, hysteresis)
        self.assertTrue(scale.item() == init_scale)

        # No infs for growth_interval-1 iterations, scale shouldn't change
        found_inf.zero_()
        for i in range(growth_interval-1):
            update_scale_hysteresis(scale, growth_tracker, hysteresis_tracker,
                    found_inf, growth_factor, backoff_factor, growth_interval, hysteresis)
        self.assertTrue(scale.item() == init_scale)

        # Infs appear for more than hysteresis iterations, scale should be backed off
        found_inf.fill_(1)
        extra_iters = random.randint(0, 1000)
        scale_before = scale.detach().item()
        scale_ref = scale_before
        for i in range(hysteresis + extra_iters):
            update_scale_hysteresis(scale, growth_tracker, hysteresis_tracker,
                    found_inf, growth_factor, backoff_factor, growth_interval, hysteresis)
        for i in range(1 + extra_iters):
            # Scale is continuously backed off for each iteration with an inf
            scale_new = scale_ref * backoff_factor
            if isfinite(scale_new):
                scale_ref = scale_new
            else:
                scale_ref = 0 # Scale update kernel does not check for underflow when backing off, which results in zero
        self.assertTrue(scale.item() == scale_ref)

        # No infs for more than growth_interval iterations, scale should be increased
        found_inf.fill_(0)
        extra_iters = random.randint(0, 1000)
        scale_before = scale.detach().item()
        scale_ref = scale_before
        for i in range(growth_interval + extra_iters):
            update_scale_hysteresis(scale, growth_tracker, hysteresis_tracker,
                    found_inf, growth_factor, backoff_factor, growth_interval, hysteresis)
        for i in range(1 + int(math.floor(extra_iters / growth_interval))):
            # Scale is grown every growth_interval iterations
            scale_new = scale_ref * growth_factor
            if isfinite(scale_new):
                scale_ref = scale_new
        self.assertTrue(scale.item() == scale_ref)


    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_fuzz(self):
        init_scale_list = [1, 1024, 65536]
        growth_factor_list = [1.0, 2.0, 4.0]
        backoff_factor_list = [0.5, 0.25]
        growth_interval_list = [10, 100]
        hysteresis_list = [10, 100]

        for init_scale in init_scale_list:
            for growth_factor in growth_factor_list:
                for backoff_factor in backoff_factor_list:
                    for growth_interval in growth_interval_list:
                        for hysteresis in hysteresis_list:
                            self.update_scale_hysteresis_body(init_scale, growth_factor,
                                    backoff_factor, growth_interval, hysteresis)



if __name__ == '__main__':
    unittest.main()
