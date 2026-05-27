import torch

from apex._custom_ops import load_custom_op_library, scalar_float


load_custom_op_library("_focal_loss_cuda", __file__)


def forward(
    cls_output,
    cls_targets_at_level,
    num_positives_sum,
    num_real_classes,
    alpha,
    gamma,
    smoothing_factor,
):
    return torch.ops.apex.focal_loss_forward(
        cls_output,
        cls_targets_at_level,
        num_positives_sum,
        num_real_classes,
        scalar_float(alpha),
        scalar_float(gamma),
        scalar_float(smoothing_factor),
    )


backward = torch.ops.apex.focal_loss_backward
