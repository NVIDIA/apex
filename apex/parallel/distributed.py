import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable

def flat_dist_call(tensors, call, extra_args=None):
    flat_dist_call.warn_on_half = True
    buckets = {}
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
                    
    if flat_dist_call.warn_on_half:
        if torch.cuda.HalfTensor in buckets:
            print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                  " It is recommended to use the NCCL backend in this case.")
            flat_dist_call.warn_on_half = False

    for tp in buckets:
        bucket = buckets[tp]
        coalesced = _flatten_dense_tensors(bucket)
        if extra_args is not None:
            call(coalesced, *extra_args)
        else:
            call(coalesced)
        if call is dist.all_reduce:
            coalesced /= dist.get_world_size()
        for buf, synced in zip(bucket, _unflatten_dense_tensors(coalesced, bucket)):
            buf.copy_(synced)
            


class DistributedDataParallel(Module):

    """
    :class:`DistributedDataParallel` is a simpler version of upstream :class:`
    DistributedDataParallel`. Its usage is designed to be used in conjunction with
    apex.parallel.multiproc.py. It assumes that your run is using multiprocess with
    1 GPU/process, that the model is on the correct device, and that
    torch.set_device has been used to set the device. Parameters are broadcasted
    to the other processes on initialization of DistributedDataParallel, and will be
    allreduced in buckets durring the backward pass.
    
    See https://github.com/csarofeen/examples/tree/apex/distributed for detailed usage.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (Default = 10000000): Minimum number of elements in a communication bucket.
    
    
    """

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

        self.module = module
        param_list = [param for param in self.module.state_dict().values() if torch.is_tensor(param)]
        if dist._backend == dist.dist_backend.NCCL:
            for param in param_list:
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."
                
        #broadcast parameters
        flat_dist_call(param_list, dist.broadcast, (0,) )

        #all reduce gradient hook
        def allreduce_params():
            if(self.needs_reduction):
                self.needs_reduction = False
            else:
                return
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)
            
        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)


    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)

    '''
    def _sync_buffers(self):
        buffers = list(self.module._all_buffers())
        if len(buffers) > 0:
            # cross-node buffer sync
            flat_buffers = _flatten_dense_tensors(buffers)
            dist.broadcast(flat_buffers, 0)
            for buf, synced in zip(buffers, _unflatten_dense_tensors(flat_buffers, buffers)):
                buf.copy_(synced)
     def train(self, mode=True):
        # Clear NCCL communicator and CUDA event cache of the default group ID,
        # These cache will be recreated at the later call. This is currently a
        # work-around for a potential NCCL deadlock.
        if dist._backend == dist.dist_backend.NCCL:
            dist._clear_group_cache()
        super(DistributedDataParallel, self).train(mode)
        self.module.train(mode)
    '''
