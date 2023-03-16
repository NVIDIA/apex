from typing import Union, Iterable

import torch

_kernel_import_succeeded = False
try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier
    _kernel_import_succeeded = True
except ImportError:
    _kernel_import_succeeded = False

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    This is identical to torch.nn.utils.clip_grad_norm_, except it
    uses a fused CUDA kernel when computing the 2-norm of GPU tensors
    in float32 and float16.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Trivial case
    if len(parameters) == 0:
        return torch.tensor(0.)

    # Fallback implementation
    if not (_kernel_import_succeeded
            and norm_type == 2.0
            and any(p.is_cuda for p in parameters)):
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite = error_if_nonfinite,
        )

    # Find fp32 and fp16 gradients on GPU
    device = next(p.device for p in parameters if p.is_cuda)
    grads_fp32, grads_fp16, grads_misc = [], [], []
    for p in parameters:
        grad = p.grad.detach()
        if p.dtype == torch.float32 and p.device == device:
            grads_fp32.append(grad)
        elif p.dtype == torch.float16 and p.device == device:
            grads_fp16.append(grad)
        else:
            grads_misc.append(grad)

    # Compute gradient L2 norms
    norms = []
    dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device=device)
    if grads_fp32:
        norms.append(
            multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads_fp32],
                False,
            )[0]
        )
    if grads_fp16:
        norms.append(
            multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads_fp16],
                False,
            )[0],
        )
    for g in grads_misc:
        norms.append(torch.linalg.norm(g).unsqueeze(0).to(device))
    total_norm = torch.linalg.norm(torch.cat(norms))

    # Check for non-finite values
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')

    # Scale gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    if grads_fp32:
        multi_tensor_applier(
            amp_C.multi_tensor_scale,
            dummy_overflow_buf,
            [grads_fp32, grads_fp32],
            clip_coef_clamped,
        )
    if grads_fp16:
        multi_tensor_applier(
            amp_C.multi_tensor_scale,
            dummy_overflow_buf,
            [grads_fp16, grads_fp16],
            clip_coef_clamped,
        )
    for g in grads_misc:
        g.mul_(clip_coef_clamped.to(g.device))

    return total_norm
