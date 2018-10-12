import torch

from .distributed import DistributedDataParallel, Reducer
try:
    import syncbn
    print("using fused syncBN")
    from .optimized_sync_batchnorm import SyncBatchNorm
except ImportError:
    print("using non-fused syncBN, try install apex with 'python setup.py install --cuda_ext' to enable fused syncBN for better performance")
    from .sync_batchnorm import SyncBatchNorm

def convert_syncbn_model(module):
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
        mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_syncbn_model(child))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod
