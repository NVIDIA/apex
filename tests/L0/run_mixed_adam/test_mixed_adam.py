import unittest
import os
import random

import torch
import apex

class TestFusedAdam(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.cuda.manual_seed(9876)

    def tearDown(self):
        pass

    def gen_param_optim(self, tensors, adam_option):
        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))

        ref_optim = torch.optim.Adam(ref_param, **adam_option)
        tst_optim = apex.optimizers.FusedAdam(tst_param, **adam_option)
       
        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
        half_grads = []
        for p_ref, p_tst in zip(ref_param, tst_param):
            half_grads.append(torch.rand_like(p_ref).half())
            p_ref.grad = half_grads[-1].float() / scale
        return half_grads

    def get_max_diff(self, ref_param, tst_param):
        max_abs_diff = max_rel_diff = 0
        for p_ref, p_tst in zip(ref_param, tst_param):
            max_abs_diff_p = (p_ref - p_tst).abs().max().item()
            max_rel_diff_p = ((p_ref - p_tst) / p_ref).abs().max().item()

            if max_abs_diff_p > max_abs_diff:  max_abs_diff = max_abs_diff_p
            if max_rel_diff_p > max_rel_diff:  max_rel_diff = max_rel_diff_p

        return max_abs_diff, max_rel_diff

    def gen_single_type_test(self, param_type=torch.float):
        nelem = 278011
        adam_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}

        tensor = torch.rand(nelem, dtype=param_type, device='cuda')
        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_param_optim([tensor], adam_option)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()
            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_double(self):
        self.gen_single_type_test(param_type=torch.double)

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    def test_half(self):
        nelem = 278011
        adam_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}

        tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_param_optim([tensor], adam_option)

        for i in range(self.iters):
            half_grads = self.gen_mixed_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step(grads=half_grads)
            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]
        adam_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}

        tensors = []
        for size in sizes:
            tensors.append(torch.rand(size, dtype=torch.float, device='cuda'))
        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_param_optim(tensors, adam_option)

        for i in range(self.iters):
            half_grads = self.gen_mixed_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step(grads=half_grads)
            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_scale(self):
        nelem = 278011
        adam_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}

        tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_param_optim([tensor], adam_option)

        for i in range(self.iters):
            scale = random.random() * 1000
            half_grads = self.gen_mixed_grad(ref_param, tst_param, scale)
            ref_optim.step()
            tst_optim.step(grads=half_grads, scale=scale)
            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_fp16_output(self):
        nelem = 278011
        adam_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}

        tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_param_optim([tensor], adam_option)

        fp16_param = torch.nn.Parameter(tensor.clone().half())

        for i in range(self.iters):
            half_grads = self.gen_mixed_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step(grads=half_grads, output_params=[fp16_param])

            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

            max_abs_diff, max_rel_diff = self.get_max_diff(tst_param, \
                [fp16_param.float()])
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_adam_option(self):
        nelem = 1
        adam_option = {'lr':0.01, 'betas':(0.6, 0.9), 'eps':3e-06,
            'weight_decay':0, 'amsgrad':False}

        tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_param_optim([tensor], adam_option)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()
            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)


if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()
