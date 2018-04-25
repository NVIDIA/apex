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
    DistributedDataParallel` that is optimized for use with NCCL. Its usage is designed
    to be used in conjunction with apex.parallel.multiproc.py. It assumes that your run
    is using multiprocess with 1 GPU/process, that the model is on the correct device,
    and that torch.set_device has been used to set the device. Parameters are broadcasted
    to the other processes on initialization of DistributedDataParallel, and will be
    allreduced in buckets durring the backward pass.
    
    See https://github.com/csarofeen/examples/tree/apex/distributed for detailed usage.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (Default = 10000000): Minimum number of elements in a communication bucket.
    
    
    """

    def __init__(self, module, message_size=10000000):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False
        
        self.message_size = message_size
        
        #reference to last iterations parameters to see if anything has changed
        self.param_refs = []
        
        self.reduction_stream = torch.cuda.Stream()
        
        self.module = module
        self.param_list = list(self.module.parameters())
        
        if dist._backend == dist.dist_backend.NCCL:
            for param in self.param_list:
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."
                
        self.record = []
        self.create_hooks()

        flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,) )
        
    def create_hooks(self):
        #all reduce gradient hook
        def allreduce_params():
            if(self.needs_reduction):
                self.needs_reduction = False
                self.needs_refresh = False
            else:
                return
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)
            t_record = torch.cuda.IntTensor(self.record)
            dist.broadcast(t_record, 0)
            self.record = [int(entry) for entry in t_record]


        def flush_buckets():
            if not self.needs_reduction:
                return
            self.needs_reduction = False
            
            ready = []
            for i in range(len(self.param_state)):
                if self.param_state[i] == 1:
                    param = self.param_list[self.record[i]]
                    if param.grad is not None:
                        ready.append(param.grad.data)

            if(len(ready)>0):
                orig_stream = torch.cuda.current_stream()
                with torch.cuda.stream(self.reduction_stream):
                    self.reduction_stream.wait_stream(orig_stream)
                    flat_dist_call(ready, dist.all_reduce)
                    
            torch.cuda.current_stream().wait_stream(self.reduction_stream)

        for param_i, param in enumerate(list(self.module.parameters())):
            def wrapper(param_i):
                
                def allreduce_hook(*unused):
                    if self.needs_refresh:
                        self.record.append(param_i)
                        Variable._execution_engine.queue_callback(allreduce_params)
                    else:
                        Variable._execution_engine.queue_callback(flush_buckets)
                        self.param_state[self.record.index(param_i)] = 1
                        self.comm_ready_buckets()
                    
                    
                if param.requires_grad:
                    param.register_hook(allreduce_hook)
            wrapper(param_i)


    def comm_ready_buckets(self):

        ready = []
        counter = 0

        while counter < len(self.param_state) and self.param_state[counter] == 2:
            counter += 1

        while counter < len(self.param_state) and self.param_state[counter] == 1:
            ready.append(counter)
            counter += 1

        if not ready:
            return

        grads = []
        for ind in ready:
            param_ind = self.record[ind]
            if self.param_list[param_ind].grad is not None:
                grads.append(self.param_list[param_ind].grad.data)

        bucket = []
        bucket_inds = []
        while grads:
            bucket.append(grads.pop(0))
            bucket_inds.append(ready.pop(0))
            
            cumm_size = 0
            for ten in bucket:
                cumm_size += ten.numel()

            if cumm_size < self.message_size:
                continue

            evt = torch.cuda.Event()
            evt.record(torch.cuda.current_stream())
            evt.wait(stream=self.reduction_stream)
        
            with torch.cuda.stream(self.reduction_stream):
                flat_dist_call(bucket, dist.all_reduce)

            for ind in bucket_inds:
                self.param_state[ind] = 2
        
    def forward(self, *inputs, **kwargs):
        """
        Forward function for DDP.
        Args:
            inputs: inputs that match the module's passed in for initialization.
            kwargs: kwargs that match the module's passed in for initialization. 

        """

        param_list = [param for param in list(self.module.parameters()) if param.requires_grad]

        

        self.needs_refresh = True if not self.param_refs else any(
            [param1 is not param2 for param1, param2 in zip(param_list, self.param_refs)]
        )
                
        if  self.needs_refresh:
            self.record = []

            
        self.param_state = [0 for i in range(len(param_list))]
        self.param_refs = param_list
        self.needs_reduction = True
        
        return self.module(*inputs, **kwargs)
