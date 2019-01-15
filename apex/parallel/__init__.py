import torch

if hasattr(torch.distributed, 'ReduceOp'):
    ReduceOp = torch.distributed.ReduceOp
elif hasattr(torch.distributed, 'reduce_op'):
    ReduceOp = torch.distributed.reduce_op
else:
    ReduceOp = torch.distributed.deprecated.reduce_op

from .distributed import DistributedDataParallel, Reducer
try:
    import syncbn
    from .optimized_sync_batchnorm import SyncBatchNorm
except ImportError:
    try:
        _ = warned_syncbn
    except NameError:
        print("Warning:  apex was installed without --cuda_ext. Fused syncbn kernels will be unavailable.  Python fallbacks will be used instead.")
        warned_syncbn = True
    from .sync_batchnorm import SyncBatchNorm

def convert_syncbn_model(module, process_group=None, channel_last=False):
    '''
    Recursively traverse module and its children to replace all
    `torch.nn.modules.batchnorm._BatchNorm` with `apex.parallel.SyncBatchNorm`

    All `torch.nn.BatchNorm*N*d` wraps around
    `torch.nn.modules.batchnorm._BatchNorm`, this function let you easily switch
    to use sync BN.

    Args:
        module: input module `torch.nn.Module`

    Examples::
        >>> # model is an instance of torch.nn.Module
        >>> import apex
        >>> sync_bn_model = apex.parallel.convert_syncbn_model(model)
    '''
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group, channel_last=channel_last)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_syncbn_model(child,
                                                  process_group=process_group,
                                                  channel_last=channel_last))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod
