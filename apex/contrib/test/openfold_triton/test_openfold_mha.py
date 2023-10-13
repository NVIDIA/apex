import math
import random
from typing import Optional
import torch
import unittest

SKIP_TEST = None
try:
    from apex.contrib.openfold_triton import AttnTri as openfold_attention_triton
except ImportError as e:
    SKIP_TEST = e


def openfold_attention_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
) -> torch.Tensor:
    # query:  [*, num_heads, Q, c_hidden]
    # key:    [*, num_heads, K, c_hidden]
    # value:  [*, num_heads, V, c_hidden]
    # mask:   Logit mask tensor broadcastable to [*, num_heads, Q, K]
    # bias:   Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
    # inf:    Safe infinity value.
    # assuming K == V

    key = torch.swapdims(key, -2, -1)
    # key: [*, num_heads, c_hidden, K]

    scaling = 1.0 / math.sqrt(query.size(-1))
    a = torch.matmul(query * scaling, key)
    # a: [*, num_heads, Q, K]

    a += (mask - 1.0) * inf
    # a: [*, num_heads, Q, K]

    if bias is not None:
        a += bias
    # a: [*, num_heads, Q, K]

    a = torch.softmax(a, dim=-1)
    # a: [*, num_heads, Q, K]

    a = torch.matmul(a, value)
    # a: [*, num_heads, Q, c_hidden]

    return a


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class OpenfoldMhaTest(unittest.TestCase):
    def setUp(self, seed=1234):
        super().setUp()
        random.seed(seed)
        torch.manual_seed(seed)

    # representative workload in openfold
    def test_openfold_triton_mha(
        self, Z=256, H=4, N_CTX=256, D_HEAD=32, dtype=torch.float16
    ):
        One = 1
        q = (
            torch.empty((One, Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0.1, std=0.2)
            .requires_grad_()
        )
        k = (
            torch.empty((One, Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0.4, std=0.2)
            .requires_grad_()
        )
        v = (
            torch.empty((One, Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0.3, std=0.2)
            .requires_grad_()
        )
        bias = (
            torch.empty((One, One, H, N_CTX, N_CTX), dtype=dtype, device="cuda")
            .normal_(mean=0.2, std=0.2)
            .requires_grad_()
        )
        mask = (
            torch.empty((One, N_CTX, One, One, N_CTX), device="cuda").normal_(
                mean=0, std=0.5
            )
            > 0
        )
        mask = mask.to(device=torch.device("cuda"), dtype=dtype).requires_grad_(False)

        dout = torch.randn_like(q)
        inf = 1e9

        # reference implementation
        ref_out = openfold_attention_eager(q, k, v, mask, bias, inf)
        ref_out.backward(dout)

        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
        ref_dbias, bias.grad = bias.grad.clone(), None

        # triton implementation
        tri_out = openfold_attention_triton(
            q, k, v, mask, bias, inf, torch.is_grad_enabled()
        )
        tri_out.backward(dout)

        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None
        tri_dbias, bias.grad = bias.grad.clone(), None

        # check results
        torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
        torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=0)
        torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=0)
        torch.testing.assert_close(ref_dbias, tri_dbias, atol=1e-2, rtol=0)


if __name__ == "__main__":
    unittest.main()

