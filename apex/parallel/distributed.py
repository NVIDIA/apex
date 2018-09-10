import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from collections import OrderedDict
import copy

def flat_dist_call(tensors, call, extra_args=None):
    flat_dist_call.warn_on_half = True
    buckets = OrderedDict()
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

            

def extract_tensors(maybe_tensor, tensor_list):
    if torch.is_tensor(maybe_tensor):
        tensor_list.append(maybe_tensor)
    else:
        try:
            for item in maybe_tensor:
                extract_tensors(item, tensor_list)
        except TypeError:
            return

        
class Reducer(object):
    """
    :class:`apex.parallel.Reducer` is a simple class that helps allreduce a module's parameters
    across processes.  :class:`Reducer` is intended to give the user additional control:
    Unlike :class:`DistributedDataParallel`, :class:`Reducer` will not automatically allreduce
    parameters during ``backward()``.
    Instead, :class:`Reducer` waits for the user to call `<reducer_instance>.reduce()` manually.
    This enables, for example, delaying the allreduce to be carried out every 
    several iterations instead of every single iteration.

    Like :class:`DistributedDataParallel`, :class:`Reducer` averages any tensors it allreduces 
    over the number of participating processes.

    :class:`Reducer` is designed to work with the launch utility script 
    ``apex.parallel.multiproc.py`` or the upstream launch utility script 
    ``torch.distributed.launch`` with --nproc_per_node <= the number of gpus per node.
    When used with these launchers, :class:`apex.parallel.multiproc.py` 
    assumes 1:1 mapping of processes to GPUs.

    main_reducer.py in https://github.com/NVIDIA/apex/tree/master/examples/imagenet shows example usage.

    Args:
        module_or_grads_list: Either a network definition (module) being run in 
        multi-gpu/distributed mode, or an iterable of gradients to be reduced. 
        If a module is passed in, the Reducer constructor will sync the parameters across processes
        (broadcasting from rank 0) to make sure they're all initialized with the same values.  
        If a list of gradients (that came from some module) 
        is passed in, the user is responsible for manually syncing that module's parameters
        at the beginning of training.
    """
    
    def __init__(self, module_or_grads_list):
        if isinstance(module_or_grads_list, Module):
            self.module = module_or_grads_list
            flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,) )

        else:
            self.module = None
            self.grads = []
            extract_tensors(module_or_grads_list, self.grads)
            
    def reduce(self):
        if self.module:
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)
        else:
            flat_dist_call(self.grads, dist.all_reduce)
            
            
class DistributedDataParallel(Module):
    """
    :class:`apex.parallel.DistributedDataParallel` is a module wrapper that enables
    easy multiprocess distributed data parallel training, similar to ``torch.nn.parallel.DistributedDataParallel``.

    :class:`DistributedDataParallel` is designed to work with
    the launch utility script ``apex.parallel.multiproc.py``.  
    When used with ``multiproc.py``, :class:`DistributedDataParallel` 
    assigns 1 process to each of the available (visible) GPUs on the node.
    Parameters are broadcast across participating processes on initialization, and gradients are
    allreduced and averaged over processes during ``backward()``.

    :class:`DistributedDataParallel` is optimized for use with NCCL.  It achieves high performance by 
    overlapping communication with computation during ``backward()`` and bucketing smaller gradient
    transfers to reduce the total number of transfers required.

    :class:`DistributedDataParallel` assumes that your script accepts the command line 
    arguments "rank" and "world-size."  It also assumes that your script calls
    ``torch.cuda.set_device(args.rank)`` before creating the model.

    https://github.com/NVIDIA/apex/tree/master/examples/distributed shows detailed usage.
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet shows another example
    that combines :class:`DistributedDataParallel` with mixed precision training.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (Default = 1e7): Minimum number of elements in a communication bucket.
        shared_param (Default = False): If your model uses shared parameters this must be True.  It will disable bucketing of parameters to avoid race conditions.

    """

    def __init__(self, module, message_size=10000000, shared_param=False):
        super(DistributedDataParallel, self).__init__()

        # Backward/forward compatibility around 
        # https://github.com/pytorch/pytorch/commit/540ef9b1fc5506369a48491af8a285a686689b36 
        if(hasattr(dist, "get_backend")):
            self._backend = dist.get_backend()
            self.backend_enum_holder = dist.DistBackend
        else:
            self._backend = dist._backend 
            self.backend_enum_holder = dist.dist_backend

        self.warn_on_half = True if self._backend == self.backend_enum_holder.GLOO else False
        self.shared_param = shared_param
        self.message_size = message_size
        
        # Will hold [param for param in self.module.parameters() if param.requires_grad]
        # aka, the active paramters this iteration.  The ordering of this list will be 
        # the same across all processes.
        self.active_params = []
        
        self.reduction_stream = torch.cuda.Stream()
        
        self.module = module
        
        if self._backend == self.backend_enum_holder.NCCL:
            for param in self.module.parameters():
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."
                
        self.record = []
        self.create_hooks()

        flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,) )


    def __setstate__(self, state):
        super(DistributedDataParallel, self).__setstate__(state)
        self.reduction_stream = torch.cuda.Stream()


    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        if self._backend != self.backend_enum_holder.NCCL:
            del attrs['self.reduction_stream']
            return attrs
        
        
    def create_hooks(self):
        # all reduce gradient hook
        def allreduce_params():
            if not self.needs_reduction:
                return
            self.needs_reduction = False

            # parameter ordering refresh
            if self.needs_refresh and not self.shared_param:
                t_record = torch.cuda.IntTensor(self.record)
                dist.broadcast(t_record, 0)
                self.record = [int(entry) for entry in t_record]
                # As before, self.record stores a list of indexes into self.active_params.
                # param_id_to_record_i is a map from each active param's id to its slot in
                # self.record.
                self.param_id_to_record_i = {id(self.active_params[a]) : i 
                                             for i, a in enumerate(self.record)}
                self.needs_refresh = False

            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)

        def flush_buckets():
            if not self.needs_reduction:
                return
            self.needs_reduction = False

            grads = []
            for i in range(self.ready_end, len(self.param_state)):
                param = self.active_params[self.record[i]]
                if param.grad is not None:
                    grads.append(param.grad.data)
            grads = [param.grad.data for param in self.ready_params] + grads

            if(len(grads)>0):
                orig_stream = torch.cuda.current_stream()
                with torch.cuda.stream(self.reduction_stream):
                    self.reduction_stream.wait_stream(orig_stream)
                    flat_dist_call(grads, dist.all_reduce)
                    
            torch.cuda.current_stream().wait_stream(self.reduction_stream)

        for param in self.module.parameters():
            if param.requires_grad:
                def wrapper(param):

                    def allreduce_hook(*unused):
                        if self.needs_refresh:
                            self.record.append(self.param_id_to_active_i[id(param)])
                            Variable._execution_engine.queue_callback(allreduce_params)
                        else:
                            Variable._execution_engine.queue_callback(flush_buckets)
                            # param_id_to_record_i handily enables us to replace the 
                            # O(N) self.record.index(param_i) call with an O(1) dict lookup. 
                            self.comm_ready_buckets(self.param_id_to_record_i[id(param)])
                        
                    param.register_hook(allreduce_hook)

                wrapper(param)


    def comm_ready_buckets(self, record_i):

        if self.param_state[record_i] != 0:
            raise RuntimeError("Error: Your model uses shared parameters, DDP flag shared_params must be set to True in initialization.")
            
        if self.param_state[self.ready_end] == 0:
            self.param_state[record_i] = 1
            return

        while self.ready_end < len(self.param_state) and self.param_state[self.ready_end] == 1:
            self.ready_params.append(self.active_params[self.record[self.ready_end]])
            self.ready_numel += self.ready_params[-1].numel()
            self.ready_end += 1

        if self.ready_numel < self.message_size:
            self.param_state[record_i] = 1
            return
            
        grads = [param.grad.data for param in self.ready_params]

        bucket = []
        bucket_inds = []
        while grads:
            bucket.append(grads.pop(0))
            
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

            for i in range(self.ready_start, self.ready_start+len(bucket)):
                self.param_state[i] = 2
                self.ready_params.pop(0)

        self.param_state[record_i] = 1

        
    def forward(self, *inputs, **kwargs):

        param_list = [param for param in self.module.parameters() if param.requires_grad]

        # Conditions under which to refresh self.record
        # Forward has the authority to set needs_refresh to True, but only allreduce_params
        # in backward has the authority to set needs_refresh to False.
        # Parentheses are not necessary for correct order of operations, but make the intent clearer.
        if ( (not self.active_params) or 
             self.shared_param or 
             (len(param_list) != len(self.active_params)) or
             any([param1 is not param2 for param1, param2 in zip(param_list, self.active_params)]) ):
            self.needs_refresh = True

        if self.needs_refresh:
            self.record = []
            # Map from each param's id to its index in the list of active parameters.
            self.param_id_to_active_i = {id(param) : i for i, param in enumerate(param_list)}  
            
        self.param_state = [0 for i in range(len(param_list))]
        self.active_params = param_list
        self.needs_reduction = True

        self.ready_start = 0
        self.ready_end   = 0
        self.ready_params = []
        self.ready_numel = 0
        
        return self.module(*inputs, **kwargs)
