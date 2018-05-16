
# TODO: think about the following two. They do weird things.
# - torch.nn.utils.clip_grad (but it should always be fp32 anyway)
# - torch.nn.utils.weight_norm

# Notes:
# F.instance_norm uses batch_norm internally. Which correctly handles
#   fp16 in/out with fp32 weights. So we shouldn't do anything for
#   either of these.
# F.normalize calls `input.norm()` internally, so it's redundant, but
#   kept here in case impl. changes.
# F.cosine_similarity is same: calls `x.norm()` internally.

import torch.nn.functional

MODULE = torch.nn.functional

FP16_FUNCS = [
    'conv1d',
    'conv2d',
    'conv3d',
    'conv_transpose1d',
    'conv_transpose2d',
    'conv_transpose3d',
    'conv_tbc', # Undocumented / maybe new?
    'linear',
]

FP32_FUNCS = [
    # Pointwise
    'softplus',
    'softmin',
    'log_softmax',
    'softmax',

    # Normalization
    'layer_norm',
    'group_norm',
    'local_response_norm',
    'normalize',
    'cosine_similarity',

    # Loss functions
    # TODO: which of these can be fp16?
    'binary_cross_entropy',
    'poisson_nll_loss',
    'cosine_embedding_loss',
    'cross_entropy',
    'hinge_embedding_loss',
    'kl_div',
    'l1_loss',
    'mse_loss',
    'margin_ranking_loss',
    'multilabel_margin_loss',
    'multilabel_soft_margin_loss',
    'multi_margin_loss',
    'nll_loss',
    'binary_cross_entropy_with_logits',
    'smooth_l1_loss',
    'soft_margin_loss',
    'triplet_margin_loss'
]
