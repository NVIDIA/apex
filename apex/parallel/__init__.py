import torch

if hasattr(torch.distributed, 'ReduceOp'):
    ReduceOp = torch.distributed.ReduceOp
elif hasattr(torch.distributed, 'reduce_op'):
    ReduceOp = torch.distributed.reduce_op
else:
    ReduceOp = torch.distributed.deprecated.reduce_op

from .distributed import DistributedDataParallel, Reducer
# This is tricky because I'd like SyncBatchNorm to be exposed the same way
# for both the cuda-enabled and python-fallback versions, and I don't want
# to suppress the error information.
try:
    import syncbn
    from .optimized_sync_batchnorm import SyncBatchNorm
except ImportError as err:
    from .sync_batchnorm import SyncBatchNorm
    SyncBatchNorm.syncbn_import_error = err

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
