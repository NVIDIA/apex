import torch
from torch.optim import Optimizer
import apex
import unittest

from test_fused_optimizer import TestFusedOptimizer
from itertools import product


class LANS(Optimizer):
    """
    Implements LANS algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
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
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)
        super(LANS, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data /= p.grad.data.norm().add(group['eps'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LANS does not support sparse gradients, consider SparseAdam instad.')

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
                m_t.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
                v_t.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Debiasing
                m_t_hat = m_t / (1.0 - beta1 ** state['step'])
                v_t_hat = v_t / (1.0 - beta2 ** state['step'])

                update_m = m_t_hat / v_t_hat.sqrt().add(group['eps'])
                update_c = grad / v_t_hat.sqrt().add(group['eps'])

                if group['weight_decay'] != 0:
                    update_m.add_(p.data, alpha=group['weight_decay'])
                    update_c.add_(p.data, alpha=group['weight_decay'])

                trust_m_ratio = 1.0
                trust_c_ratio = 1.0
                w_norm = p.data.pow(2).sum().sqrt()
                m_norm = update_m.pow(2).sum().sqrt()
                c_norm = update_c.pow(2).sum().sqrt()
                if w_norm > 0 and m_norm > 0:
                    trust_m_ratio = w_norm / m_norm
                if w_norm > 0 and c_norm > 0:
                    trust_c_ratio = w_norm / c_norm

                step_size = group['lr']

                p.data.add_(update_m, alpha=-step_size * beta1 * trust_m_ratio)
                p.data.add_(update_c, alpha=-step_size * (1 - beta1) * trust_c_ratio)

        return loss


class TestFusedLANS(TestFusedOptimizer):

    def __init__(self, *args, **kwargs):
        super(TestFusedLANS, self).__init__(*args, **kwargs)

        # The options for LANS and FusedLANS are very specific if they
        # are expected to behave the same.
        self.options = {'lr': 1e-3, 'betas': (0.95, 0), 'eps': 1e-8,
                        'weight_decay': 0}

        self.tst_options = {'lr': 1e-3, 'betas': (0.95, 0), 'eps': 1e-8,
                            'weight_decay': 0}

        self.ref_optim = LANS
        self.fused_optim = apex.optimizers.FusedLANS

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16)

    @unittest.skipIf(torch.cuda.device_count() < 2, "more than 1 GPU required")
    def test_multi_device(self):
        devices = ("cuda:1", "cuda:0")
        for current_dev, tensor_dev in product(devices, devices):
            with torch.cuda.device(current_dev):
                torch.cuda.synchronize()
                self.gen_single_type_test(param_type=torch.float, device=tensor_dev)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]

        tensors = []
        for size in sizes:
            tensors.append(torch.rand(size, dtype=torch.float, device="cuda"))
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
            tensors, self.options, self.tst_options
        )

        for _ in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()
            max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    @unittest.skipIf(~torch.cuda.is_bf16_supported() == False, "bfloat16 is not supported")
    def test_stochastic_rounding(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]

        tensors = []
        for size in sizes:
            tensors.append(torch.rand(size, dtype=torch.bfloat16, device="cuda"))
        tst_options = {'lr': 1e-3, 'betas': (0.95, 0), 'eps': 1e-8,
                       'weight_decay': 0, 'stochastic_rounding': True}
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
            tensors, self.options, tst_options
        )

        for _ in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            tst_optim.step()


if __name__ == '__main__':
    unittest.main()
