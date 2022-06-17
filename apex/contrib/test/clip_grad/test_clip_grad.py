import random
import unittest

import torch
from apex.contrib.clip_grad import clip_grad_norm_

def make_params(
        num_params,
        sizes=[1,2,3,4,5],
        num_dims=[1,2,3],
        dtypes=[torch.float32],
        devices=['cuda'],
        make_copy=False,
):
    """Construct parameters with random configurations"""

    # Construct parameters
    params = []
    for _ in range(num_params):
        dims = [random.choice(sizes) for _ in range(random.choice(num_dims))]
        dtype = random.choice(dtypes)
        device = random.choice(devices)
        p = torch.nn.Parameter(torch.randn(dims, dtype=dtype, device=device))
        p.grad = torch.randn_like(p)
        params.append(p)

    # Copy parameters if needed
    if make_copy:
        params_copy = []
        for p in params:
            p_copy = p.clone().detach()
            p_copy.grad = p.grad.clone().detach()
            params_copy.append(p_copy)
        return params, params_copy
    else:
        return params

class ClipGradNormTest(unittest.TestCase):

    def setUp(self, seed=1234):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def test_matches_pytorch(
            self,
            num_params=41,
            dtypes=[torch.float32, torch.float16, torch.float64],
            devices=['cuda', 'cpu'],
            max_norm=0.54321,
            norm_type=2.0,
            rtol=1e-3,
    ):
        """Make sure PyTorch and Apex gradient clipping produce same results"""
        torch_params, apex_params = make_params(
            num_params,
            dtypes=dtypes,
            devices=devices,
            make_copy=True,
        )
        torch_norm = torch.nn.utils.clip_grad_norm_(
            torch_params,
            max_norm,
            norm_type=norm_type,
        )
        apex_norm = clip_grad_norm_(
            apex_params,
            max_norm,
            norm_type=norm_type,
        )
        self.assertTrue(
            torch.allclose(
                torch_norm.to(dtype=torch.float64),
                apex_norm.to(dtype=torch.float64),
                rtol=rtol,
            ),
            msg=('Discrepancy in gradient norms: '
                 f'torch = {torch_norm.item()}, apex = {apex_norm.item()})'))
        for torch_p, apex_p in zip(torch_params, apex_params):
            torch_g = torch_p.grad.to(dtype=torch.float64, device='cpu')
            apex_g = apex_p.grad.to(dtype=torch.float64, device='cpu')
            g_rel_err = torch.max(torch.abs((torch_g-apex_g)/(torch_g+1e-32)))
            self.assertTrue(
                torch.equal(torch_p, apex_p),
                msg=f'Torch and Apex parameters are not identical')
            self.assertTrue(
                torch.allclose(torch_g, apex_g, rtol=rtol),
                msg=f'Discrepancy in gradients: relative error = {g_rel_err}')

    def test_matches_pytorch_fp16(self):
        self.test_matches_pytorch(num_params=11, dtypes=[torch.float16])

    def test_matches_pytorch_fp32(self):
        self.test_matches_pytorch(dtypes=[torch.float32], rtol=1e-6)

    def test_matches_pytorch_fp64(self):
        self.test_matches_pytorch(dtypes=[torch.float64], rtol=1e-15)

    def test_matches_pytorch_cpu(self):
        self.test_matches_pytorch(devices=['cpu'])

    def test_matches_pytorch_infnorm(self):
        self.test_matches_pytorch(norm_type=float('inf'))

    def test_matches_pytorch_1norm(self):
        self.test_matches_pytorch(norm_type=1.0)

    def test_raises_on_nan(self):
        params = make_params(5, num_dims=[1])
        params[2].grad[-1] = float('NaN')
        self.assertRaises(
            RuntimeError, clip_grad_norm_, params, 1.0, error_if_nonfinite=True)

    def test_raises_on_inf(self):
        params = make_params(5, num_dims=[1])
        params[2].grad[-1] = float('inf')
        self.assertRaises(
            RuntimeError, clip_grad_norm_, params, 1.0, error_if_nonfinite=True)

if __name__ == "__main__":
    unittest.main()
