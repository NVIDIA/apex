"""Test for fused RoPE functions.

Ref: https://github.com/NVIDIA/Megatron-LM/blob/40becfc96c4144985458ac0e0fae45dbb111fbd2/megatron/fused_kernels/tests/test_fused_kernels.py
"""  # NOQA
import itertools

import torch
from torch.testing._internal import common_utils
from apex.transformer.functional import fused_apply_rotary_pos_emb


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Copied from Megatron-Core for testing.
# https://github.com/NVIDIA/Megatron-LM/blob/5f2877d85cb26e47ce6dcdae4b80adf376abf4e8/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L139
def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


class TestFusedRoPE(common_utils.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.head_num = 64
        self.seq_length = [2048, 4096]
        self.hidden_size = [128, 256]
        self.rotary_percent = [0.5, 1.0]
        self.dtype = [torch.float32, torch.bfloat16, torch.float16]
        self.memory_format = [torch.contiguous_format, torch.channels_last]
        self.loss_func = [self._overlapping_grad, self._non_overlapping_grad]
        self.device = torch.cuda.current_device()

    def tearDown(self) -> None:
        torch.cuda.empty_cache()
        super().tearDown()

    def _overlapping_grad(self, output) -> torch.Tensor:
        return output.sum() * 2

    def _non_overlapping_grad(self, output) -> torch.Tensor:
        t = torch.ones_like(output)
        return torch.sum(output * t)

    def test_forward_backward(self):
        for (
            dtype,
            seq_length,
            hidden_size,
            rotary_percent,
            memory_format,
            loss_func,
        ) in itertools.product(
            self.dtype,
            self.seq_length,
            self.hidden_size,
            self.rotary_percent,
            self.memory_format,
            self.loss_func,
        ):
            t = torch.rand(
                (seq_length, self.batch_size, self.head_num, hidden_size),
                dtype=dtype,
                device=self.device,
            )
            t = t.contiguous(memory_format=memory_format)
            t.requires_grad = True

            emb = torch.rand(
                (seq_length, 1, 1, int(hidden_size * rotary_percent)),
                dtype=torch.float32,
                device=self.device,
            )

            # unfused
            output_unfused = apply_rotary_pos_emb(t, emb)
            loss_unfused = loss_func(output_unfused)
            loss_unfused.backward()
            grad_unfused = t.grad.detach().clone()
            t.grad = None

            # fused
            output_fused = fused_apply_rotary_pos_emb(t, emb)
            loss_fused = loss_func(output_fused)
            loss_fused.backward()
            grad_fused = t.grad.detach().clone()
            t.grad = None

            self.assertEqual(
                output_unfused,
                output_fused,
                msg=f"{dtype=}, {seq_length=}, {hidden_size=}, {rotary_percent=}, {memory_format=}, loss_func={loss_func.__name__}",
            )
            self.assertEqual(
                grad_unfused,
                grad_fused,
                msg=f"{dtype=}, {seq_length=}, {hidden_size=}, {rotary_percent=}, {memory_format=}, loss_func={loss_func.__name__}",
            )


if __name__ == "__main__":
    common_utils.run_tests()
