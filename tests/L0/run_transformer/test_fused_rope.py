"""Test for fused RoPE functions.

Ref: https://github.com/NVIDIA/Megatron-LM/blob/40becfc96c4144985458ac0e0fae45dbb111fbd2/megatron/fused_kernels/tests/test_fused_kernels.py
"""  # NOQA

import itertools

import torch
from torch.testing._internal import common_utils
from apex.transformer.functional import (
    fused_apply_rotary_pos_emb,
    fused_apply_rotary_pos_emb_cached,
    fused_apply_rotary_pos_emb_thd,
    fused_apply_rotary_pos_emb_2d,
)


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


def apply_rotary_pos_emb_thd(
    t: torch.Tensor, cu_seqlens: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            apply_rotary_pos_emb(x.unsqueeze(1), freqs[: x.size(0)])
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def apply_rotary_pos_emb_2d(q, img_h, img_w, cos_h, sin_h, cos_w, sin_w):
    q = q.view(q.shape[0], img_h, img_w, q.shape[2], q.shape[3])
    q1, q2 = q.chunk(2, dim=-1)
    cos_h = cos_h[:, :img_h].unsqueeze(2)  # [1, H, 1, 1, D//2]
    sin_h = sin_h[:, :img_h].unsqueeze(2)  # [1, H, 1, 1, D//2]
    q1 = (q1 * cos_h) + (_rotate_half(q1) * sin_h)
    cos_w = cos_w[:, :img_w].unsqueeze(1)  # [1, 1, W, 1, D//2]
    sin_w = sin_w[:, :img_w].unsqueeze(1)  # [1, 1, W, 1, D//2]
    q2 = (q2 * cos_w) + (_rotate_half(q2) * sin_w)
    return torch.cat([q1, q2], dim=-1).view(q.shape[0], -1, q.shape[3], q.shape[4])


class TestFusedRoPE(common_utils.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.head_num = 64
        self.seq_length = [2048, 4096]
        self.hidden_size = [128, 256]
        self.rotary_percent = [0.5, 1.0]
        self.dtype = [torch.float32, torch.bfloat16, torch.float16]
        self.transpose = [None, (0, 1), (2, 3)]
        self.transpose_output_memory = [False, True]
        self.loss_func = [self._overlapping_grad, self._non_overlapping_grad]
        self.cached = [False, True]
        self.device = torch.cuda.current_device()
        # for 2D RoPE
        self.img_h = [32, 64]
        self.img_w = [32, 64]

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
            transpose,
            transpose_output_memory,
            loss_func,
            cached,
        ) in itertools.product(
            self.dtype,
            self.seq_length,
            self.hidden_size,
            self.rotary_percent,
            self.transpose,
            self.transpose_output_memory,
            self.loss_func,
            self.cached,
        ):
            t = torch.rand(
                (seq_length, self.batch_size, self.head_num, hidden_size),
                dtype=dtype,
                device=self.device,
            )
            if transpose:
                t = t.transpose(*transpose).contiguous().transpose(*transpose)
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
            if cached:
                cos, sin = emb.cos(), emb.sin()
                output_fused = fused_apply_rotary_pos_emb_cached(
                    t, cos, sin, transpose_output_memory=transpose_output_memory
                )
            else:
                output_fused = fused_apply_rotary_pos_emb(
                    t, emb, transpose_output_memory=transpose_output_memory
                )
            loss_fused = loss_func(output_fused)
            loss_fused.backward()
            grad_fused = t.grad.detach().clone()
            t.grad = None

            self.assertEqual(
                output_unfused,
                output_fused,
                msg=f"{dtype=}, {seq_length=}, {hidden_size=}, {rotary_percent=}, "
                f"{transpose=}, {transpose_output_memory=}, loss_func={loss_func.__name__}",
            )
            self.assertEqual(
                grad_unfused,
                grad_fused,
                msg=f"{dtype=}, {seq_length=}, {hidden_size=}, {rotary_percent=}, "
                f"{transpose=}, {transpose_output_memory=}, loss_func={loss_func.__name__}",
            )
            assert (
                output_fused.transpose(0, 1).is_contiguous() is transpose_output_memory
            )

    def test_thd_forward_backward(self):
        cu_seqlens = torch.tensor(
            [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048],
            dtype=torch.int32,
            device=self.device,
        )
        for (
            dtype,
            hidden_size,
            rotary_percent,
            transpose,
            loss_func,
        ) in itertools.product(
            self.dtype,
            self.hidden_size,
            self.rotary_percent,
            [None, [1, 2]],
            self.loss_func,
        ):
            t = torch.rand(
                (cu_seqlens[-1], self.head_num, hidden_size),
                dtype=dtype,
                device=self.device,
            )
            if transpose:
                t = t.transpose(*transpose).contiguous().transpose(*transpose)
            t.requires_grad = True

            emb = torch.rand(
                (cu_seqlens[-1], 1, 1, int(hidden_size * rotary_percent)),
                dtype=torch.float32,
                device=self.device,
            )

            # unfused
            output_unfused = apply_rotary_pos_emb_thd(t, cu_seqlens, emb)
            loss_unfused = loss_func(output_unfused)
            loss_unfused.backward()
            grad_unfused = t.grad.detach().clone()
            t.grad = None

            # fused
            output_fused = fused_apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                emb,
            )
            loss_fused = loss_func(output_fused)
            loss_fused.backward()
            grad_fused = t.grad.detach().clone()
            t.grad = None

            self.assertEqual(
                output_unfused,
                output_fused,
                msg=f"{dtype=}, {cu_seqlens=}, {hidden_size=}, {rotary_percent=}, "
                f"{transpose=}, loss_func={loss_func.__name__}",
            )
            self.assertEqual(
                grad_unfused,
                grad_fused,
                msg=f"{dtype=}, {cu_seqlens=}, {hidden_size=}, {rotary_percent=}, "
                f"{transpose=}, loss_func={loss_func.__name__}",
            )

    def test_2d_forward_backward(self):
        for (
            dtype,
            img_h,
            img_w,
            hidden_size,
            transpose,
            loss_func,
            margin,
        ) in itertools.product(
            self.dtype,
            self.img_h,
            self.img_w,
            self.hidden_size,
            self.transpose,
            self.loss_func,
            [0, 3],
        ):
            t = torch.rand(
                (self.batch_size, img_h * img_w, self.head_num, hidden_size),
                dtype=dtype,
                device=self.device,
            )
            if transpose:
                t = t.transpose(*transpose).contiguous().transpose(*transpose)
            t.requires_grad = True

            emb_h = torch.rand(
                (1, img_h + margin, 1, hidden_size // 2),
                dtype=torch.float32,
                device=self.device,
            )
            cos_h, sin_h = emb_h.cos().to(dtype), emb_h.sin().to(dtype)

            emb_w = torch.rand(
                (1, img_w + margin, 1, hidden_size // 2),
                dtype=torch.float32,
                device=self.device,
            )
            cos_w, sin_w = emb_w.cos().to(dtype), emb_w.sin().to(dtype)

            # unfused
            output_unfused = apply_rotary_pos_emb_2d(
                t, img_h, img_w, cos_h, sin_h, cos_w, sin_w
            )
            loss_unfused = loss_func(output_unfused)
            loss_unfused.backward()
            grad_unfused = t.grad.detach().clone()
            t.grad = None

            # fused
            output_fused = fused_apply_rotary_pos_emb_2d(
                t, img_h, img_w, cos_h, sin_h, cos_w, sin_w
            )
            loss_fused = loss_func(output_fused)
            loss_fused.backward()
            grad_fused = t.grad.detach().clone()
            t.grad = None

            self.assertEqual(
                output_unfused,
                output_fused,
                msg=f"{dtype=}, {img_h=}, {img_w=}, {hidden_size=}, "
                f"{transpose=}, loss_func={loss_func.__name__}",
            )
            self.assertEqual(
                grad_unfused,
                grad_fused,
                msg=f"{dtype=}, {img_h=}, {img_w=}, {hidden_size=}, "
                f"{transpose=}, loss_func={loss_func.__name__}",
            )


if __name__ == "__main__":
    common_utils.run_tests()
