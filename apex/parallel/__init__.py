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
    Designed to work with apex sync BN
    replaces all BN layer in the model with sync BN
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for child in module.children():
        mod.add_module(convert_syncbn_model(child))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod
