import random
import unittest

import torch

SKIP_TEST = None
try:
    from apex.contrib.clip_grad import clip_grad_norm_
except ImportError as e:
    SKIP_TEST = e


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


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class ClipGradNormTest(unittest.TestCase):

    def setUp(self, seed=1234):
        super().setUp()
        random.seed(seed)
        torch.manual_seed(seed)

    def test_matches_pytorch(
            self,
            num_params=41,
            dtypes=[torch.float32, torch.float16, torch.float64],
            devices=['cuda', 'cpu'],
            max_norm=0.54321,
            norm_type=2.0,
            rtol=1e-3,
            atol=1e-20,
    ):
        """Make sure PyTorch and Apex gradient clipping produce same results"""

        # Construct identical sets of parameters
        torch_params, apex_params = make_params(
            num_params,
            dtypes=dtypes,
            devices=devices,
            make_copy=True,
        )

        # Apply gradient clipping
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

        # Make sure PyTorch and Apex get same results
        torch.testing.assert_close(
            apex_norm, torch_norm,
            rtol=rtol,
            atol=atol,
            check_dtype=False,
        )
        for torch_p, apex_p in zip(torch_params, apex_params):
            torch.testing.assert_close(
                apex_p, torch_p,
                rtol=0,
                atol=0,
            ) # Params should be unaffected
            torch.testing.assert_close(
                apex_p.grad, torch_p.grad,
                rtol=rtol,
                atol=atol,
            )

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

    def test_raises_on_mismatch(self):

        # Construct different sets of parameters
        torch_params, apex_params = make_params(7, make_copy=True)
        with torch.no_grad():
            torch_params[0].grad.view(-1)[0] = 1.23
            apex_params[0].grad.view(-1)[0] = 3.21

        # Apply gradient clipping
        torch_norm = torch.nn.utils.clip_grad_norm_(
            torch_params,
            0.54321,
        )
        apex_norm = clip_grad_norm_(
            apex_params,
            0.54321,
        )

        # Make sure PyTorch and Apex get different results
        self.assertRaises(
            AssertionError,
            torch.testing.assert_close,
            apex_norm, torch_norm,
            rtol=1e-3,
            atol=1e-20,
            check_dtype=False,
        )
        for torch_p, apex_p in zip(torch_params, apex_params):
            self.assertRaises(
                AssertionError,
                torch.testing.assert_close,
                apex_p.grad, torch_p.grad,
                rtol=1e-3,
                atol=1e-20,
            )

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
