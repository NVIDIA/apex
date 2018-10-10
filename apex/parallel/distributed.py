import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from collections import OrderedDict
from itertools import chain
import copy

# apply_dist_call requires that tensors in 'bucket' are all the same type.
def apply_flat_dist_call(bucket, call, extra_args=None):
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
        apply_flat_dist_call(bucket, call, extra_args)

            
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
        module_or_grads_list: Either a network definition (module) being run in multi-gpu/distributed mode, or an iterable of gradients to be reduced.  If a module is passed in, the Reducer constructor will sync the parameters across processes (broadcasting from rank 0) to make sure they're all initialized with the same values.  If a list of gradients (that came from some module) is passed in, the user is responsible for manually syncing that module's parameters at the beginning of training.
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
        delay_allreduce (Default = False):  Delay all communication to the end of the backward pass.  This disables overlapping communication with computation.

    """

    def __init__(self, 
                 module, 
                 message_size=10000000, 
                 delay_allreduce=False, 
                 shared_param=None,
                 allreduce_trigger_params=None):
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

        if shared_param is not None:
            raise ValueError("shared_param is no longer supported as an option.  It was misleadingly named from the start.  It turns out overlapping communication with computation should work fine with shared parameters.  If you still wish to delay communication to the end of the backward pass, use delay_allreduce=True|False instead.") 

        self.custom_allreduce_triggers = False
        if allreduce_trigger_params is not None:
            if delay_allreduce:
                raise ValueError("Setting allreduce_trigger_params is only valid if delay_allreduce=False.")  
            self.custom_allreduce_triggers = True
            self.allreduce_trigger_params = set([id(param) for param in allreduce_trigger_params])

        self.delay_allreduce = delay_allreduce
        self.message_size = message_size

        self.reduction_stream = torch.cuda.Stream()
        
        self.module = module
        
        if self._backend == self.backend_enum_holder.NCCL:
            for param in self.module.parameters():
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."

        self.active_params = []

        self.param_type_to_tmp_i = {"torch.cuda.HalfTensor" : 0, 
                                    "torch.cuda.FloatTensor" : 1,
                                    "torch.cuda.DoubleTensor" : 2}
                 
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
        # Append leftover buckets
        for tmp_bucket in self.tmp_buckets:
            if len(tmp_bucket) > 0:
                self.active_i_buckets.append(tmp_bucket)

        self.num_buckets = len(self.active_i_buckets)
        self.bucket_sizes = [len(bucket) for bucket in self.active_i_buckets]

        info_tensor = torch.cuda.IntTensor([self.num_buckets] + 
                                           self.bucket_sizes + 
                                           list(chain(*self.active_i_buckets)))

        dist.broadcast(info_tensor, 0)

        info = [int(entry) for entry in info_tensor]

        self.num_buckets = info[0]
        self.bucket_sizes = info[1:self.num_buckets + 1] 
        self.buckets = [[None for _ in range(self.bucket_sizes[i])] for i in range(self.num_buckets)] 
        
        flattened_buckets = info[self.num_buckets + 1:]
        flat_i = 0
        for bucket_idx in range(self.num_buckets): 
            for bucket_loc in range(self.bucket_sizes[bucket_idx]):
                param_i = flattened_buckets[flat_i]
                self.param_id_to_bucket[id(self.active_params[param_i])] = (bucket_idx, bucket_loc)
                flat_i += 1 
        
        
    def create_hooks(self):
        # Fallback hook that's only called at the end of backward.
        # Used if you deliberately want to delay allreduces to the end, or to refresh the 
        # bucket structure that will be used to overlap communication with computation in later
        # iterations.
        def allreduce_params():
            # Bucket record refresh
            if not self.delay_allreduce:
                if self.needs_refresh:
                    self.sync_bucket_structure()

                    self.needs_refresh = False

            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
           
            flat_dist_call(grads, dist.all_reduce)

        def overlapping_backward_epilogue():
            torch.cuda.current_stream().wait_stream(self.reduction_stream)
     
            # Sanity checks that all the buckets were kicked off
            if self.next_bucket != self.num_buckets:
                raise RuntimeError("In epilogue, next_bucket != num_buckets.  "
                                   "This probably indicates some buckets were not allreduced.")

            for actual, expected in zip(self.buckets_ready_size, self.bucket_sizes):
                if actual != expected:
                    raise RuntimeError("Some param buckets were not allreduced.")
           

        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                def wrapper(param):
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]

                    def allreduce_hook(*unused):
                        if self.delay_allreduce or self.needs_refresh:
                            # TODO:  How do we want to handle multiple backward passes between
                            # each forward, e.g., backward passes with retain_graph=True?
                            # needs_refresh and callback_queued are both vulnerable states.
                            if not self.delay_allreduce and self.needs_refresh:
                                # Use the backward pass to build the bucket structure on the fly.
                                active_i = self.param_id_to_active_i[id(param)]

                                # Float, half, and double tensors are grouped into buckets separately.
                                current_type = self.param_type_to_tmp_i[param.type()]
  
                                self.tmp_buckets[current_type].append(active_i)                          

                                ship_tmp_bucket = False
                                if self.custom_allreduce_triggers:
                                    if id(param) in self.allreduce_trigger_params:
                                        ship_tmp_bucket = True
                                else:
                                    self.tmp_numels[current_type] += param.numel()
                                    if self.tmp_numels[current_type] >= self.message_size:
                                        ship_tmp_bucket = True

                                # To consider:  If custom_allreduce_triggers are in use, ship all
                                # tmp_buckets, not just tmp_buckets[current_type].
                                if ship_tmp_bucket:
                                    self.active_i_buckets.append(self.tmp_buckets[current_type])
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
        # Need to do this in every hook for compatibility with Ruberry's streaming backward PR.
        # self.reduction_stream.wait_stream(torch.cuda.current_stream())

        bucket_idx, bucket_loc = self.param_id_to_bucket[id(param)]

        if self.buckets[bucket_idx][bucket_loc] is not None:
            raise RuntimeError("The backward pass is attempting to replace an already-filled "
                               "bucket slot.  This is almost certainly an error.")

        self.buckets[bucket_idx][bucket_loc] = param.grad.data
        self.buckets_ready_size[bucket_idx] += 1

        if self.buckets_ready_size[bucket_idx] == self.bucket_sizes[bucket_idx]:
            if bucket_idx == self.next_bucket:
                self.reduction_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.reduction_stream):
                    apply_flat_dist_call(self.buckets[bucket_idx], dist.all_reduce)

                    self.next_bucket += 1

                    # Reversing upstream's logic here, because we constructed our buckets based on
                    # the order things were received during backward.
                    if len(self.ready_buckets_not_reduced) > 0:
                        sorted_todo = sorted(self.ready_buckets_not_reduced)
                        for i in sorted_todo:
                            # Nothing can be reduced now
                            if i > self.next_bucket:
                                break
                            elif i == self.next_bucket:
                                apply_flat_dist_call(self.buckets[i], dist.all_reduce)
                                self.ready_buckets_not_reduced.remove(i)
                                self.next_bucket += 1 
                            else:
                                raise ValueError("i should always be >= next_bucket")
            else:
                self.ready_buckets_not_reduced.add(bucket_idx)

        
    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        
        if not self.delay_allreduce:
            param_list = [param for param in self.module.parameters() if param.requires_grad]

            # Conditions under which to refresh self.record
            # Forward has the authority to set needs_refresh to True, but only allreduce_params
            # in backward has the authority to set needs_refresh to False.
            # Parentheses are not necessary for correct order of operations, but make the intent clearer.
            if ((not self.active_params) or 
                (len(param_list) != len(self.active_params)) or
                any([param1 is not param2 for param1, param2 in zip(param_list, self.active_params)])):
                self.needs_refresh = True

            if self.needs_refresh:
                self.active_i_buckets = []
                self.buckets = []
                self.tmp_buckets = [[], [], []] # [running half, float, double buckets]
                self.tmp_numels = [0, 0, 0]
                self.bucket_sizes = []
                self.param_id_to_active_i = {id(param) : i for i, param in enumerate(param_list)}  
                self.param_id_to_bucket = {}
            else:
                self.buckets = [[None for _ in range(self.bucket_sizes[i])] 
                                for i in range(self.num_buckets)] 
                self.buckets_ready_size = [0 for i in range(self.num_buckets)]
                self.next_bucket = 0
                self.ready_buckets_not_reduced = set()
            
            self.active_params = param_list

        self.callback_queued = False
        
        return result
