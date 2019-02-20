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
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`apex.parallel.SyncBatchNorm`.

    All ``torch.nn.BatchNorm*N*d`` wrap around
    ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
    to use sync BN.

    Args:
        module (torch.nn.Module): input module

    Example::

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

def create_syncbn_process_group(group_size):
    '''
    Creates process groups to be used for syncbn of a give ``group_size`` and returns
    process group that current GPU participates in.

    ``group_size`` must divide the total number of GPUs (world_size).

    ``group_size`` of 0 would be considered as =world_size. In this case ``None`` will be returned.

    ``group_size`` of 1 would be equivalent to using non-sync bn, but will still carry the overhead.

    Args:
        group_size (int): number of GPU's to collaborate for sync bn

    Example::

        >>> # model is an instance of torch.nn.Module
        >>> import apex
        >>> group = apex.parallel.create_syncbn_process_group(group_size)
    '''

    if group_size==0:
        return None

    world_size = torch.distributed.get_world_size()
    assert(world_size >= group_size)
    assert(world_size % group_size == 0)

    group=None
    for group_num in (range(world_size//group_size)):
        group_ids = range(group_num*group_size, (group_num+1)*group_size)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if (torch.distributed.get_rank()//group_size == group_num):
            group = cur_group
            #can not drop out and return here, every process must go through creation of all subgroups

    assert(group is not None)
    return group
