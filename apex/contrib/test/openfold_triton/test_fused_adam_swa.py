# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
import random
import unittest

import torch
import torch.nn as nn

SKIP_TEST = None
try:
    from apex.contrib.openfold_triton.fused_adam_swa import AdamMathType, FusedAdamSWA
except ImportError as e:
    SKIP_TEST = e


# Stochastic weight average (SWA) reference code from
# https://github.com/mlcommons/hpc_results_v3.0/blob/350e46f7/NVIDIA/benchmarks/openfold/implementations/pytorch/openfold/swa.py#L21-L70
class AlphaFoldSWA(nn.Module):
    """AlphaFold SWA (Stochastic Weight Averaging) module wrapper."""

    def __init__(self, alphafold: nn.Module, enabled: bool, decay_rate: float) -> None:
        super(AlphaFoldSWA, self).__init__()
        if enabled:
            self.averaged_model = torch.optim.swa_utils.AveragedModel(
                model=alphafold,
                avg_fn=swa_avg_fn(decay_rate=decay_rate),
            )
            self.enabled = True
        else:
            self.averaged_model = None
            self.enabled = False

    def update(self, alphafold: nn.Module) -> None:
        if self.enabled:
            self.averaged_model.update_parameters(model=alphafold)

    def forward(self, batch):
        if not self.enabled:
            raise RuntimeError("AlphaFoldSWA is not enabled")
        return self.averaged_model(batch)


class swa_avg_fn:
    """Averaging function for EMA with configurable decay rate
    (Supplementary '1.11.7 Evaluator setup')."""

    def __init__(self, decay_rate: float) -> None:
        self._decay_rate = decay_rate

    def __call__(
        self,
        averaged_model_parameter: torch.Tensor,
        model_parameter: torch.Tensor,
        num_averaged: torch.Tensor,
    ) -> torch.Tensor:
        # for decay_rate = 0.999:
        # return averaged_model_parameter * 0.999 + model_parameter * 0.001
        # avg * 0.999 + m * 0.001
        # 999*avg/1000 + m/1000
        # (999*avg + avg - avg)/1000 + m/1000
        # (1000*avg - avg)/1000 + m/1000
        # 1000*avg/1000 - avg/1000 + m/1000
        # avg + (m - avg)/1000
        # avg + (m - avg)*0.001
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) * (
            1.0 - self._decay_rate
        )


@unittest.skipIf(SKIP_TEST, f"Skip testing FusedAdamSWA: {SKIP_TEST}")
class FusedAdamSWATestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._seed = 19260817
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.backends.cudnn.deterministic = True

    def test_fused_update_on_random_data(self):
        device = torch.device("cuda:0")
        compute_dtype = torch.float32
        state_dtype = torch.float64
        atol = 1e-5  # Default: 1e-8, raise error at 1e-6 for FP32 compute and FP64 state.
        rtol = 1e-4  # Default: 1e-5
        lr = 1e-1
        bias_correction = True
        beta1, beta2 = 0.9, 0.999
        eps = 1e-6
        adam_math_mode = AdamMathType.PyTorchAdam
        weight_decay = 1e-3  # PyTorchAdam impl will fail non-zero weight decay.
        amsgrad = False
        adam_step = 1900
        swa_decay_rate = 0.9
        swa_n_averaged = 1

        state_params = [
            torch.empty(random.randint(128, 2048), device=device, dtype=state_dtype).uniform_(-5, 5)
            for _ in range(32)
        ]
        compute_dtypes = [
            compute_dtype if random.uniform(0.0, 1.0) <= 0.5 else state_dtype for _ in range(32)
        ]
        grads = [
            torch.empty_like(p, dtype=d).uniform_(-5, 5) for d, p in zip(compute_dtypes, state_params)
        ]
        moments = [torch.empty_like(p).uniform_(-5, 5) for p in state_params]
        velocities = [torch.empty_like(p).uniform_(0, 10) for p in state_params]

        # Ground truth: Apex FusedAdam, optimized-hpc SWA.
        compute_params_gt = [p.clone().to(d) for d, p in zip(compute_dtypes, state_params)]
        dummy_model = torch.nn.Module()
        for i, p in enumerate(state_params):
            dummy_model.register_parameter(f"param_{i}", torch.nn.Parameter(p.clone()))
        state_params_gt = list(dummy_model.parameters())
        swa_model = AlphaFoldSWA(dummy_model, enabled=True, decay_rate=swa_decay_rate)
        swa_params_gt = list(swa_model.parameters())
        optimizer = torch.optim.Adam(
            state_params_gt,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        moments_gt, velocities_gt = [], []
        for i, p in enumerate(optimizer.param_groups[0]["params"]):
            s = optimizer.state[p]
            self.assertTrue(moments[i].shape == velocities[i].shape == p.shape)
            s["step"] = torch.tensor(adam_step, dtype=state_dtype, device=device)
            s["exp_avg"] = moments[i].clone()
            s["exp_avg_sq"] = velocities[i].clone()
            moments_gt.append(s["exp_avg"])
            velocities_gt.append(s["exp_avg_sq"])
        for p, g in zip(state_params_gt, grads):
            p.grad = g.clone().to(state_dtype)
        optimizer.step()
        swa_model.averaged_model.n_averaged.copy_(swa_n_averaged)
        swa_model.update(dummy_model)
        for c, s in zip(compute_params_gt, state_params_gt):
            c.detach().copy_(s.detach().to(c.dtype))

        # Fused AdamSWA, all at once.
        state_params_test = [torch.nn.Parameter(p.clone()) for p in state_params]
        compute_params_test = [p.clone().to(d) for d, p in zip(compute_dtypes, state_params)]
        swa_params_test = [p.clone() for p in state_params]
        fused_optimizer = FusedAdamSWA(
            params=state_params_test,
            compute_params=compute_params_test,
            swa_params=swa_params_test,
            swa_decay_rate=swa_decay_rate,
            lr=lr,
            bias_correction=bias_correction,
            betas=(beta1, beta2),
            eps=eps,
            adam_math_mode=adam_math_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        moments_test, velocities_test = [], []
        for i, p in enumerate(fused_optimizer.param_groups[0]["params"]):
            s = fused_optimizer.state[p]
            self.assertTrue(moments[i].shape == velocities[i].shape == p.shape)
            s["exp_avg"] = moments[i].clone()
            s["exp_avg_sq"] = velocities[i].clone()
            moments_test.append(s["exp_avg"])
            velocities_test.append(s["exp_avg_sq"])
        for c, g in zip(compute_params_test, grads):
            c.grad = g.clone()
        fused_optimizer.param_groups[0]["step"] = adam_step
        fused_optimizer.swa_param_groups[0]["n_averaged"] = swa_n_averaged
        fused_optimizer.step()

        # Ensure parameters are actually updated.
        for i, (p_gt, p_test, p_origin) in enumerate(
            zip(state_params_gt, state_params_test, state_params)
        ):
            self.assertFalse(torch.allclose(p_gt, p_origin, rtol=rtol, atol=atol))
            self.assertFalse(torch.allclose(p_test, p_origin, rtol=rtol, atol=atol))
        # Ensure FusedAdamSWA correctness.
        self.assertEqual(
            swa_model.averaged_model.n_averaged.item(),
            fused_optimizer.swa_param_groups[0]["n_averaged"]
        )
        for i, (p_test, p_gt) in enumerate(
            zip(
                chain(state_params_test, compute_params_test, swa_params_test),
                chain(state_params_gt, compute_params_gt, swa_params_gt),
            )
        ):
            self.assertTrue(torch.allclose(p_test, p_gt, rtol=rtol, atol=atol))
        # Ensure moments are updated correctly.
        for i, (m, m_gt) in enumerate(
            zip(
                chain(moments_test, velocities_test),
                chain(moments_gt, velocities_gt),
            )
        ):
            self.assertTrue(torch.allclose(m, m_gt, rtol=rtol, atol=atol))


if __name__ == "__main__":
    unittest.main()
