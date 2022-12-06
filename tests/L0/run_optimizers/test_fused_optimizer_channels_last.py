from itertools import product
import random
import unittest

import torch

import apex

# NHWC
class TestFusedOptimizerChannelsLast(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.manual_seed(9876)

    def tearDown(self):
        pass

    def gen_param_optim(self, tensors, options, device, tst_options=None):

        # Adding this to make backward compatible with existing tests. Just in
        # case "tst_options" are not provided, it gets a copy of options
        # which contains the parameters for the reference optimizer
        if tst_options == None:
            tst_options = options

        ref_param = []
        tst_param = []
        for tensor in tensors:
            input = tensor.clone().contiguous(memory_format=torch.channels_last).to(device) # channels_last
            ref_input = tensor.clone().contiguous().to(device)

            self.assertTrue(input.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_input.is_contiguous(memory_format=torch.contiguous_format))

            tst_param.append(torch.nn.Parameter(input))
            ref_param.append(torch.nn.Parameter(ref_input))

        ref_optim = self.ref_optim(ref_param, **options)
        tst_optim = self.fused_optim(tst_param, **tst_options)
        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad.clone() #### p_tst is =torch.channels_last but p_tst.grad is torch.contiguous_format

            self.assertTrue(p_tst.grad.is_contiguous(memory_format=torch.contiguous_format))
            self.assertTrue(p_ref.grad.is_contiguous(memory_format=torch.contiguous_format))


    def get_max_diff(self, ref_param, tst_param):
        max_abs_diff = max_rel_diff = 0
        for p_ref, p_tst in zip(ref_param, tst_param):
            self.assertTrue(p_ref.is_contiguous(memory_format=torch.contiguous_format))
            self.assertTrue(p_tst.is_contiguous(memory_format=torch.channels_last))
            max_abs_diff_p = (p_ref - p_tst).abs().max().item()
            max_rel_diff_p = ((p_ref - p_tst) / p_ref).abs().max().item()

            if max_abs_diff_p > max_abs_diff:  max_abs_diff = max_abs_diff_p
            if max_rel_diff_p > max_rel_diff:  max_rel_diff = max_rel_diff_p

        return max_abs_diff, max_rel_diff

    def gen_single_type_test(self, param_type=torch.float, device='cuda', *, skip_assert: bool = False):
        # nelem = 278011

        # Some ref and test optimizers may require different set of options.
        # This is a quick workaround to add that functionality while making
        # minimum changes in existing code.
        # If there is no "tst_options" field provided, safe to initialize
        # the test optimizer with the parameters of reference optimizer.
        if not hasattr(self, 'tst_options'):
            self.tst_options = self.options

        tensor = torch.rand([3,4,2,3], dtype=param_type, device=device)
        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_param_optim([tensor], self.options, device, self.tst_options)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()
            if skip_assert:
                return
            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

class TestFusedSGDChannelLast(TestFusedOptimizerChannelsLast):
    def __init__(self, *args, **kwargs):
        super(TestFusedSGDChannelLast, self).__init__(*args, **kwargs)
        self.options = {"lr": .25, "momentum": .125}
        self.ref_optim = torch.optim.SGD
        self.fused_optim = apex.optimizers.FusedSGD

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16)

    @unittest.skipIf(torch.cuda.device_count()<2, "more than 1 GPU required")
    def test_multi_device(self):
        devices = ("cuda:0", "cuda:1")
        for current_dev, tensor_dev in product(devices, devices):
            with torch.cuda.device(current_dev):
                self.gen_single_type_test(param_type=torch.float, device=tensor_dev)

if __name__ == '__main__':
    unittest.main()
