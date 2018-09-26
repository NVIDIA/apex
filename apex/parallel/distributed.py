import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from collections import OrderedDict
from itertools import chain
import copy

# apply_dist_call requires that tensors in 'bucket' are all the same type.
def apply_flat_dist_call(bucket, extra_args=None):
    coalesced = _flatten_dense_tensors(bucket)

    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)

    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()
        
    for buf, synced in zip(bucket, _unflatten_dense_tensors(coalesced, bucket)):
        buf.copy_(synced)


# flat_dist_call organizes 'tensors' by type.
def flat_dist_call(tensors, call, extra_args=None):
    flat_dist_call.warn_on_half = True
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
                    
    for tp in buckets:
        bucket = buckets[tp]
        apply_flat_dist_call(bucket, extra_args)

            
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

        self.reduction_stream = torch.cuda.Stream()
        
        self.module = module
        
        if self._backend == self.backend_enum_holder.NCCL:
            for param in self.module.parameters():
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."

        self.active_params = []
                 
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
      
    # Broadcast rank 0's bucket structure across all processes, and have all processes 
    # regenerate their bucket structures to match. 
    def sync_bucket_structure(self):
        self.num_buckets = len(buckets)
        self.bucket_sizes = [len(bucket) for bucket in self.buckets]
        info_tensor = torch.cuda.IntTensor([self.num_buckets] + self.bucket_sizes + list(chain(*buckets)))

        dist.broadcast(info_tensor, 0)

        info = [int(entry) for entry in info_tensor]

        self.num_buckets = info[0]
        self.bucket_sizes = a[1:self.num_buckets + 1] 
        self.buckets = [[None for _ in range(bucket_sizes[i])] for i in range(self.num_buckets)] 
        
        flattened_buckets = info[self.num_buckets + 1:]
        flat_i = 0
        for bucket_i in range(self.num_buckets): 
            for bucket_loc in range(self.bucket_sizes[bucket_i])
                param_i = flattened_buckets[flat_i]
                self.param_id_to_bucket[id(self.active_params[param_i])] = (bucket_i, bucket_loc)
                flat_i += 1 
        
        
    def create_hooks(self):
        # Fallback hook that's only called at the end of backward.
        # Used if you deliberately want to delay allreduces to the end, or to refresh the 
        # bucket structure that will be used to overlap communication with computation in later
        # iterations.
        def allreduce_params():
            # Bucket record refresh
            if self.needs_refresh and not self.shared_param:

                # Append leftover buckets
                for tmp_bucket in self.tmp_buckets:
                    if len(tmp_bucket) > 0:
                        self.buckets.append(tmp_bucket)

                self.sync_bucket_structure()

                self.needs_refresh = False

            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
           
            flat_dist_call(grads, dist.all_reduce)

        def overlapping_backward_epilogue(self):
            torch.cuda.current_stream().wait_stream(self.reduction_stream)

        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                def wrapper(param):

                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]

                    def allreduce_hook(*unused):
                        if self.needs_refresh or self.shared_param:
                            if not self.shared_param:
                                # Use the backward pass to build the bucket structure on the fly.
                                active_i = self.param_id_to_active_i[id(param)]

                                # FloatTensors and HalfTensors are grouped into buckets separately.
                                if param.type() == "torch.cuda.HalfTensor": 
                                    current_type = 0
                                elif param.type == "torch.cuda.FloatTensor":
                                    current_type = 1
                                else:
                                    raise TypeError("param is neither cuda float nor cuda half.")
  
                                self.tmp_buckets[current_type].append(active_i)                           
                                self.tmp_numels[current_type] += param.numel()

                                if self.tmp_numels[current_type] > message_size:
                                    self.buckets.append(tmp_buckets[current_type])
                                    self.tmp_buckets[current_type] = []
                                    self.tmp_numels[current_type] = 0
                            
                            if not self.callback_queued:
                                Variable._execution_engine.queue_callback(allreduce_params)
                                self.callback_queued = True
                        else:
                            if not self.callback_queued:
                                Variable._execution_engine.queue_callback(overlapping_backward_epilogue)
                                self.callback_queued = True 

                            self.comm_ready_buckets(param)
                        
                    grad_acc.register_hook(allreduce_hook)
                    self.grad_accs.append(grad_acc)

                wrapper(param)


    def comm_ready_buckets(self, param):
        torch.cuda.nvtx.range_push("comm_ready_buckets")

        # Need to do this in every hook for compatibility with Ruberry's streaming backward PR.
        self.reduction_stream.wait_stream(torch.cuda.current_stream())

        bucket, bucket_loc = self.param_id_to_bucket[id(param)]

        self.buckets[bucket][bucket_loc] = param.grad
        self.buckets_ready_size[bucket] += 1

        if self.buckets_ready_size[bucket] == self.bucket_sizes[bucket]:
            if bucket == self.next_bucket:
                with torch.cuda.stream(self.reduction_stream):
                    apply_flat_dist_call(buckets[bucket], dist.all_reduce)

                self.next_bucket -= 1

                # Borrowing upstream's logic here.  The closer I can keep our logic to theirs, the
                # easier it will be to merge eventually.
                if len(self.ready_buckets_not_reduced > 0):
                    sorted_todo = sorted(self.ready_buckets_not_reduced, reverse=True)
                    for i in sorted_todo:
                        # Nothing can be reduced now
                        if i < self.next_bucket:
                            break
                        elif i == self.next_bucket:
                            with torch.cuda.stream(self.reduction_stream):
                                apply_flat_dist_call(buckets[i], dist.all_reduce)
                            self.ready_buckets_not_reduced.remove(i)
                            self.next_bucket -= 1 
                        else:
                            raise ValueError("i should always be <= next_bucket")
            else:
                self.ready_buckets_not_reduced.add(bucket)

        torch.cuda.nvtx.range_pop()

        
    def forward(self, *inputs, **kwargs):
        if not self.shared_param:
            param_list = [param for param in self.module.parameters() if param.requires_grad]

            # Conditions under which to refresh self.record
            # Forward has the authority to set needs_refresh to True, but only allreduce_params
            # in backward has the authority to set needs_refresh to False.
            # Parentheses are not necessary for correct order of operations, but make the intent clearer.
            if ( (not self.active_params) or 
                 (len(param_list) != len(self.active_params)) or
                 any([param1 is not param2 for param1, param2 in zip(param_list, self.active_params)]) ):
                self.needs_refresh = True

            if self.needs_refresh:
                self.buckets = [[]]
                self.tmp_buckets = [[], []] # [running half bucket, running float bucket]
                self.tmp_numels = [0, 0]
                self.bucket_sizes = []
                self.param_id_to_active_i = {id(param) : i for i, param in enumerate(param_list)}  
            else:
                self.buckets = [[None for _ in range(bucket_sizes[i])] for i in range(self.num_buckets)] 
                self.buckets_ready_size = [0 for i in range(self.num_buckets)]
                self.next_bucket = self.num_buckets
                self.ready_buckets_not_reduced = set()
            
            self.active_params = param_list

        self.callback_queued = False
        
        return self.module(*inputs, **kwargs)
