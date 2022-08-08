import unittest
import os

import torch
from torch.optim import Optimizer
import apex
from apex.multi_tensor_apply import multi_tensor_applier
from itertools import product

class RefLAMB(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RefLAMB, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_lamb = amp_C.multi_tensor_lamb
        else:
            raise RuntimeError('apex.optimizers.FusedLAMB requires cuda extensions')

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32, fp16, and bf16 params
        g_all_32, g_all_16, g_all_bf16 = [], [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                elif p.dtype == torch.bfloat16:
                    g_all_bf16.append(p.grad.data)
                else:
                    raise RuntimeError('FusedLAMB only support fp16, fp32, and bf16.')

        device = self.param_groups[0]["params"][0].device
        g_norm_32, g_norm_16, g_norm_bf16 = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = multi_tensor_applier(self.multi_tensor_l2norm,
                                             self._dummy_overflow_buf,
                                             [g_all_32], False)[0]
        if len(g_all_16) > 0:
            g_norm_16 = multi_tensor_applier(self.multi_tensor_l2norm,
                                             self._dummy_overflow_buf,
                                             [g_all_16], False)[0]
        if len(g_all_bf16) > 0:
            g_norm_bf16 = multi_tensor_applier(self.multi_tensor_l2norm,
                                             self._dummy_overflow_buf,
                                             [g_all_bf16], False)[0]

        # blend two grad norms to get global grad norm
        global_grad_norm = multi_tensor_applier(self.multi_tensor_l2norm,
                                                self._dummy_overflow_buf,
                                                [[g_norm_32, g_norm_16, g_norm_bf16]],
                                                False)[0]

        max_grad_norm = 1.0
        clipped_ratio = max_grad_norm / max(global_grad_norm, max_grad_norm)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data *= clipped_ratio
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)

                m_t, v_t = state['m'], state['v']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # m_t = beta1 * m + (1 - beta1) * g_t
                m_t.mul_(beta1).add_(grad, alpha=1-beta1)
                # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
                if len(g_all_16) > 0:
                    v_t.mul_(beta2)
                    v_t = v_t.to(torch.float32)
                    grad32 = grad.to(torch.float32)
                    v_t.addcmul_(grad32, grad32, value=1-beta2)
                else:
                    v_t.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Debiasing
                m_t_hat = m_t / (1.0 - beta1 ** state['step'])
                v_t_hat = v_t / (1.0 - beta2 ** state['step'])

                update = m_t_hat / v_t_hat.sqrt().add(group['eps'])

                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                trust_ratio = 1.0
                w_norm = p.data.to(torch.float32).pow(2).sum().sqrt()
                g_norm = update.pow(2).sum().sqrt()
                if w_norm > 0 and g_norm > 0:
                    trust_ratio = w_norm / g_norm

                state['w_norm'] = w_norm
                state['g_norm'] = g_norm
                state['trust_ratio'] = trust_ratio

                step_size = group['lr']

                p.data.add_(update, alpha=-step_size*trust_ratio)

        return loss

class TestLamb(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.cuda.manual_seed(9876)


    def tearDown(self):
        pass

    def gen_param_optim(self, tensors, lamb_option):
        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))

        ref_optim = self.ref_optim(ref_param, **lamb_option)
        tst_optim = self.tst_optim(tst_param, use_nvlamb=True, **lamb_option)

        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
        half_grads = []
        for p_ref, _ in zip(ref_param, tst_param):
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

    def gen_single_type_test(self, param_type=torch.float, device="cuda"):
        nelem = 18011
        tensor = torch.rand(nelem, dtype=param_type, device=device)
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08, 'weight_decay':wd}
            ref_param, tst_param, ref_optim, tst_optim = \
                self.gen_param_optim([tensor], lamb_option)

            if isinstance(tst_optim, apex.optimizers.FusedMixedPrecisionLamb):
                if param_type != torch.float:
                    # joseli: This parameter is usually passed into the constructor, 
                    # but I do not want to change the testing interface.
                    # As long as this parameter is set before the first call to step(), 
                    # then it should act normally.
                    tst_optim.reduced_precision_dtype = param_type
            for i in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                torch.cuda.synchronize()
                tst_optim.step()
                torch.cuda.synchronize()
                torch.testing.assert_close(tst_param, ref_param)

class TestFusedLAMB(TestLamb):
    def __init__(self, *args, **kwargs):
        super(TestLamb, self).__init__(*args, **kwargs)
        self.ref_optim = RefLAMB
        self.tst_optim = apex.optimizers.FusedLAMB


    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    @unittest.skip("PyTorch optimizer is not numerically correct for fp16")
    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16)

    @unittest.skipIf(torch.cuda.device_count()<2, "more than 1 GPU required")
    def test_multi_device(self):
        devices = ("cuda:0", "cuda:1")
        for current_dev, tensor_dev in product(devices, devices):
            with torch.cuda.device(current_dev):
                self.gen_single_type_test(param_type=torch.float, device=tensor_dev)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08, 'weight_decay':wd}
            tensors = []
            for size in sizes:
                tensors.append(torch.rand(size, dtype=torch.float, device='cuda'))
            ref_param, tst_param, ref_optim, tst_optim = \
                self.gen_param_optim(tensors, lamb_option)

            for i in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
                self.assertLessEqual(max_abs_diff, self.max_abs_diff)
                self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_lamb_option(self):
        nelem = 1
        tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {'lr':0.01, 'betas':(0.6, 0.9), 'eps':3e-06, 'weight_decay':wd}
            ref_param, tst_param, ref_optim, tst_optim = \
                self.gen_param_optim([tensor], lamb_option)

            for i in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

                self.assertLessEqual(max_abs_diff, self.max_abs_diff)
                self.assertLessEqual(max_rel_diff, self.max_rel_diff)

class TestFusedMixedPrecisionLamb(TestLamb):
    def __init__(self, *args, **kwargs):
        super(TestLamb, self).__init__(*args, **kwargs)
        self.ref_optim = RefLAMB
        self.tst_optim = apex.optimizers.FusedMixedPrecisionLamb


    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    def test_bfloat16(self):
        self.iters = 4
        self.gen_single_type_test(param_type=torch.bfloat16)

    def test_half(self):
        self.iters = 1
        self.gen_single_type_test(param_type=torch.float16)

    @unittest.skipIf(torch.cuda.device_count()<2, "more than 1 GPU required")
    def test_multi_device(self):
        devices = ("cuda:0", "cuda:1")
        for current_dev, tensor_dev in product(devices, devices):
            with torch.cuda.device(current_dev):
                self.gen_single_type_test(param_type=torch.float, device=tensor_dev)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08, 'weight_decay':wd}
            tensors = []
            for size in sizes:
                tensors.append(torch.rand(size, dtype=torch.float, device='cuda'))
            ref_param, tst_param, ref_optim, tst_optim = \
                self.gen_param_optim(tensors, lamb_option)

            for i in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
                self.assertLessEqual(max_abs_diff, self.max_abs_diff)
                self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_lamb_option(self):
        nelem = 1
        tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {'lr':0.01, 'betas':(0.6, 0.9), 'eps':3e-06, 'weight_decay':wd}
            ref_param, tst_param, ref_optim, tst_optim = \
                self.gen_param_optim([tensor], lamb_option)

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
