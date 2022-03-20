import torch

import focal_loss_cuda


class FocalLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cls_output,
        cls_targets_at_level,
        num_positives_sum,
        num_real_classes,
        alpha,
        gamma,
        label_smoothing=0.0,
    ):
        loss, partial_grad = focal_loss_cuda.forward(
            cls_output,
            cls_targets_at_level,
            num_positives_sum,
            num_real_classes,
            alpha,
            gamma,
            label_smoothing,
        )

        ctx.save_for_backward(partial_grad, num_positives_sum)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        partial_grad, num_positives_sum = ctx.saved_tensors

        # The backward kernel is actually in-place to save memory space,
        # partial_grad and grad_input are the same tensor.
        grad_input = focal_loss_cuda.backward(grad_loss, partial_grad, num_positives_sum)

        return grad_input, None, None, None, None, None, None


def focal_loss(
    cls_output: torch.Tensor,
    cls_targets_at_level: torch.Tensor,
    num_positive_sum: torch.Tensor,
    num_real_classes: int,
    alpha: float,
    gamma: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Fused focal loss function."""
    return FocalLoss.apply(
        cls_output,
        cls_targets_at_level,
        num_positive_sum,
        num_real_classes,
        alpha,
        gamma,
        label_smoothing,
    )
