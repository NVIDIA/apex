from collections import OrderedDict
import copy
import importlib
from itertools import chain

import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable

from ..multi_tensor_apply import multi_tensor_applier

imported_flatten_impl = False

def import_flatten_impl():
    global flatten_impl, unflatten_impl, imported_flatten_impl
    try:
        import apex_C
        flatten_impl = apex_C.flatten
        unflatten_impl = apex_C.unflatten
    except ImportError:
        print("Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten.")
        flatten_impl = torch._utils._flatten_dense_tensors
        unflatten_impl = torch._utils._unflatten_dense_tensors
    imported_flatten_impl = True

def flatten(bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return flatten_impl(bucket)

def unflatten(coalesced, bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return unflatten_impl(coalesced, bucket)

# apply_dist_call requires that tensors in 'bucket' are all the same type.
def apply_flat_dist_call(bucket, call, extra_args=None):

    coalesced = flatten(bucket)

    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)

    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()

    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
        buf.copy_(synced)

def split_half_float_double(tensors):
    dtypes = ["torch.cuda.HalfTensor",  "torch.cuda.FloatTensor", "torch.cuda.DoubleTensor"]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets

def split_by_type(tensors):
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
    return buckets

# flat_dist_call organizes 'tensors' by type.
def flat_dist_call(tensors, call, extra_args=None):
    buckets = split_by_type(tensors)

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
    Instead, :class:`Reducer` waits for the user to call ``<reducer_instance>.reduce()`` manually.
    This enables, for example, delaying the allreduce to be carried out every
    several iterations instead of every single iteration.

    Like :class:`DistributedDataParallel`, :class:`Reducer` averages any tensors it allreduces
    over the number of participating processes.

    :class:`Reducer` is designed to work with the upstream launch utility script
    ``torch.distributed.launch`` with ``--nproc_per_node <= number of gpus per node``.
    When used with this launcher, :class:`Reducer` assumes 1:1 mapping of processes to GPUs.
    It also assumes that your script calls ``torch.cuda.set_device(args.rank)`` before creating the model.

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
    easy multiprocess distributed data parallel training, similar to ``torch.nn.parallel.DistributedDataParallel``.  Parameters are broadcast across participating processes on initialization, and gradients are
    allreduced and averaged over processes during ``backward()``.

    :class:`DistributedDataParallel` is optimized for use with NCCL.  It achieves high performance by
    overlapping communication with computation during ``backward()`` and bucketing smaller gradient
    transfers to reduce the total number of transfers required.

    :class:`DistributedDataParallel` is designed to work with the upstream launch utility script
    ``torch.distributed.launch`` with ``--nproc_per_node <= number of gpus per node``.
    When used with this launcher, :class:`DistributedDataParallel` assumes 1:1 mapping of processes to GPUs.
    It also assumes that your script calls ``torch.cuda.set_device(args.rank)`` before creating the model.

    https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed shows detailed usage.
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet shows another example
    that combines :class:`DistributedDataParallel` with mixed precision training.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (int, default=1e7): Minimum number of elements in a communication bucket.
        delay_allreduce (bool, default=False):  Delay all communication to the end of the backward pass.  This disables overlapping communication with computation.
        allreduce_trigger_params (list, optional, default=None):  If supplied, should contain a list of parameters drawn from the model.  Allreduces will be kicked off whenever one of these parameters receives its gradient (as opposed to when a bucket of size message_size is full).  At the end of backward(), a cleanup allreduce to catch any remaining gradients will also be performed automatically.  If allreduce_trigger_params is supplied, the message_size argument will be ignored.
        allreduce_always_fp32 (bool, default=False):  Convert any FP16 gradients to FP32 before allreducing.  This can improve stability for widely scaled-out runs.
        gradient_average (bool, default=True):  Option to toggle whether or not DDP averages the allreduced gradients over processes.  For proper scaling, the default value of True is recommended.
        gradient_predivide_factor (float, default=1.0):  Allows perfoming the average of gradients over processes partially before and partially after the allreduce.  Before allreduce:  ``grads.mul_(1.0/gradient_predivide_factor)``.  After allreduce:  ``grads.mul_(gradient_predivide_factor/world size)``.  This can reduce the stress on the dynamic range of FP16 allreduces for widely scaled-out runs.

    .. warning::
        If ``gradient_average=False``, the pre-allreduce division (``grads.mul_(1.0/gradient_predivide_factor)``) will still be applied, but the post-allreduce gradient averaging (``grads.mul_(gradient_predivide_factor/world size)``) will be omitted.

    """

    def __init__(self,
                 module,
                 message_size=10000000,
                 delay_allreduce=False,
                 shared_param=None,
                 allreduce_trigger_params=None,
                 retain_allreduce_buffers=False,
                 allreduce_always_fp32=False,
                 num_allreduce_streams=1,
                 allreduce_communicators=None,
                 gradient_average=True,
                 gradient_predivide_factor=1.0,
                 gradient_average_split_factor=None,
                 prof=False):
        super(DistributedDataParallel, self).__init__()
        from apex import deprecated_warning
        deprecated_warning("apex.parallel.DistributedDataParallel is deprecated and will be removed by the end of February 2023.")

        # Backward/forward compatibility around
        # https://github.com/pytorch/pytorch/commit/540ef9b1fc5506369a48491af8a285a686689b36 and
        # https://github.com/pytorch/pytorch/commit/044d00516ccd6572c0d6ab6d54587155b02a3b86
        if hasattr(dist, "get_backend"):
            self._backend = dist.get_backend()
            if hasattr(dist, "DistBackend"):
                self.backend_enum_holder = dist.DistBackend
            else:
                self.backend_enum_holder = dist.Backend
        else:
            self._backend = dist._backend
            self.backend_enum_holder = dist.dist_backend

        self.warn_on_half = True if self._backend == self.backend_enum_holder.GLOO else False

        self.prof = prof

        self.allreduce_different_streams = (num_allreduce_streams > 1)
        self.num_allreduce_streams = num_allreduce_streams
        self.allreduce_communicators = allreduce_communicators
        if self.allreduce_communicators:
            assert len(allreduce_communicators[0]) == num_allreduce_streams
            assert len(allreduce_communicators[0]) == len(allreduce_communicators[1])
            assert self.allreduce_different_streams

        if self.allreduce_different_streams and delay_allreduce:
            raise ValueError("self.allreduce_different_streams may only be used if delay_allreduce=False.")

        if shared_param is not None:
            raise ValueError("shared_param is no longer supported as an option.  It was misleadingly named from the start.  It turns out overlapping communication with computation should work fine with shared parameters.  If you still wish to delay communication to the end of the backward pass, use delay_allreduce=True|False instead.")

        self.world_size = float(dist.get_world_size())

        self.retain_allreduce_buffers = retain_allreduce_buffers
        self.allreduce_always_fp32 = allreduce_always_fp32
        self.gradient_average = gradient_average
        self.gradient_predivide_factor = gradient_predivide_factor

        self.custom_allreduce_triggers = False
        if allreduce_trigger_params is not None:
            if delay_allreduce:
                raise ValueError("Setting allreduce_trigger_params is only valid if delay_allreduce=False.")
            self.custom_allreduce_triggers = True
            self.allreduce_trigger_params = set([id(param) for param in allreduce_trigger_params])

        self.delay_allreduce = delay_allreduce
        self.message_size = message_size

        self.main_stream = torch.cuda.current_stream()

        self.bucket_streams = []
        self.bucket_events = []

        self.module = module

        self._disable_allreduce = False

        if self._backend == self.backend_enum_holder.NCCL:
            for param in self.module.parameters():
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."

        self.active_params = []

        self.param_type_to_tmp_i = {"torch.cuda.HalfTensor" : 0,
                                    "torch.cuda.FloatTensor" : 1,
                                    "torch.cuda.DoubleTensor" : 2}

        if multi_tensor_applier.available:
            # TODO:  I really need to centralize the C++ backed imports
            import amp_C
            self.multi_tensor_scale = amp_C.multi_tensor_scale
            self._overflow_buf = torch.cuda.IntTensor([0])

        self.create_hooks()

        flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,) )


    def __setstate__(self, state):
        super(DistributedDataParallel, self).__setstate__(state)
        if self.allreduce_different_streams and delay_allreduce:
            raise ValueError("self.allreduce_different_streams may only be used if delay_allreduce=False.")

        if self.delay_allreduce:
            self.needs_refresh = True

        self.bucket_streams = []
        self.bucket_events = []


    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        if self._backend != self.backend_enum_holder.NCCL:
            del attrs['self.bucket_streams']
            del attrs['self.bucket_events']
            return attrs

    def enable_allreduce(self):
        self._disable_allreduce = False

    def disable_allreduce(self):
        self._disable_allreduce = True

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
        self.buckets = [[None for _ in range(self.bucket_sizes[i])]
                        for i in range(self.num_buckets)]
        # Technically, active_i_buckets' work is done.  But the information is still useful to
        # keep around.  Therefore, refresh active_i_buckets based on rank 0 as well.
        self.active_i_buckets = [[None for _ in range(self.bucket_sizes[i])]
                                 for i in range(self.num_buckets)]

        flattened_buckets = info[self.num_buckets + 1:]
        flat_i = 0
        for bucket_idx in range(self.num_buckets):
            for bucket_loc in range(self.bucket_sizes[bucket_idx]):
                param_i = flattened_buckets[flat_i]
                self.active_i_buckets[bucket_idx][bucket_loc] = param_i
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

            self.allreduce_fallback()


        def overlapping_backward_epilogue():
            for stream, event in zip(self.bucket_streams, self.bucket_events):
                stream.record_event(event)
                torch.cuda.current_stream().wait_event(event)

            # Sanity checks that all the buckets were kicked off
            if self.next_bucket != self.num_buckets:
                raise RuntimeError("In epilogue, next_bucket ({}) != num_buckets ({}).  ".format(
                                   self.next_bucket, self.num_buckets),
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
                        if self.prof:
                            torch.cuda.nvtx.range_push("allreduce_hook")

                        if not self._disable_allreduce:
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

                        if self.prof:
                            torch.cuda.nvtx.range_pop()

                    grad_acc.register_hook(allreduce_hook)
                    self.grad_accs.append(grad_acc)

                wrapper(param)


    def _stream_this_bucket(self, bucket_idx):
        if self.allreduce_different_streams:
            return self.bucket_streams[bucket_idx%self.num_allreduce_streams]
        else:
            return self.bucket_streams[0]


    def _event_this_bucket(self, bucket_idx):
        if self.allreduce_different_streams:
            return self.bucket_events[bucket_idx%self.num_allreduce_streams]
        else:
            return self.bucket_events[0]


    def allreduce_bucket(self, bucket, bucket_idx, force_default_stream):
        tensor = flatten(bucket)

        if force_default_stream:
            bucket_stream = self.main_stream
        else:
            bucket_stream = self._stream_this_bucket(bucket_idx)
            bucket_event = self._event_this_bucket(bucket_idx)
            torch.cuda.current_stream().record_event(bucket_event)
            bucket_stream.wait_event(bucket_event)

        with torch.cuda.stream(bucket_stream):
            # self.main_stream.wait_stream(torch.cuda.current_stream())
            # torch.cuda.synchronize()

            tensor_to_allreduce = tensor

            if self.allreduce_always_fp32:
                tensor_to_allreduce = tensor.float()

            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1./self.gradient_predivide_factor)

            if self.allreduce_different_streams and not force_default_stream:
                dist.all_reduce(tensor_to_allreduce, group=self.bucket_pgs[bucket_idx%self.num_allreduce_streams])
            else:
                dist.all_reduce(tensor_to_allreduce)

            if self.gradient_average:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor/self.world_size)

            if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
                tensor.copy_(tensor_to_allreduce)

            if not self.retain_allreduce_buffers:
                if multi_tensor_applier.available:
                    multi_tensor_applier(
                        self.multi_tensor_scale,
                        self._overflow_buf,
                        [unflatten(tensor, bucket), bucket],
                        1.0)
                else:
                    for buf, synced in zip(bucket, unflatten(tensor, bucket)):
                        buf.copy_(synced)

            # I think we actually do need this here.  After allreduce_bucket returns, tensor will
            # eventually go out of scope and die, at which point it could otherwise be freed for
            # further reuse by the main stream while the allreduce/div/unflatten are underway in bucket_stream.
            tensor.record_stream(bucket_stream)

        return tensor


    def allreduce_maybe_retain(self, bucket, bucket_idx, force_default_stream=False):
        allreduced = self.allreduce_bucket(bucket, bucket_idx, force_default_stream)
        if self.retain_allreduce_buffers:
            if self.allreduce_buffers[bucket_idx] is not None:
                raise RuntimeError("The backward pass is attempting to replace an already-filled "
                                   "allreduce buffer.  This is almost certainly an error.")
            self.allreduce_buffers[bucket_idx] = allreduced
            for view, grad in zip(unflatten(allreduced, bucket), bucket):
                grad.data = view
            # for buf, synced in zip(bucket, unflatten(allreduced, bucket)):
            #     buf.copy_(synced)


    def allreduce_fallback(self):
        for stream, event in zip(self.bucket_streams, self.bucket_events):
            stream.record_event(event)
            torch.cuda.current_stream().wait_event(event)

        if self.retain_allreduce_buffers:
            grads = [param.grad for param in self.module.parameters() if param.grad is not None]
        else:
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]

        split_buckets = split_half_float_double(grads)

        # If retain_allreduce_buffers is True and delay_allreduce is False,
        # this will only be done during the first backward pass, ignored by the
        # training script, and overwritten in the next forward pass.  So it's harmless.
        if self.retain_allreduce_buffers:
            self.allreduce_buffers = [None for _ in range(len(split_buckets))]

        for i, bucket in enumerate(split_buckets):
            allreduced = self.allreduce_maybe_retain(bucket, i, force_default_stream=True)


    def comm_ready_buckets(self, param):
        # Need to do this in every hook for compatibility with Ruberry's streaming backward PR.
        # self.reduction_stream.wait_stream(torch.cuda.current_stream())
        if self.prof:
            torch.cuda.nvtx.range_push("comm_ready_buckets")

        bucket_idx, bucket_loc = self.param_id_to_bucket[id(param)]

        if self.buckets[bucket_idx][bucket_loc] is not None:
            raise RuntimeError("The backward pass is attempting to replace an already-filled "
                               "bucket slot.  This is almost certainly an error.")

        if self.retain_allreduce_buffers:
            self.buckets[bucket_idx][bucket_loc] = param.grad
        else:
            self.buckets[bucket_idx][bucket_loc] = param.grad.data

        self.buckets_ready_size[bucket_idx] += 1

        if self.buckets_ready_size[bucket_idx] == self.bucket_sizes[bucket_idx]:
            if bucket_idx == self.next_bucket:
                self.allreduce_maybe_retain(self.buckets[bucket_idx], bucket_idx)

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
                            self.allreduce_maybe_retain(self.buckets[i], i)
                            self.ready_buckets_not_reduced.remove(i)
                            self.next_bucket += 1
                        else:
                            raise ValueError("i should always be >= next_bucket")
            else:
                self.ready_buckets_not_reduced.add(bucket_idx)

        if self.prof:
            torch.cuda.nvtx.range_pop()


    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)

        if self.prof:
            torch.cuda.nvtx.range_push("forward pass DDP logic")

        if not self._disable_allreduce:
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
                    self.bucket_pgs = []
                    self.bucket_streams = []
                    self.bucket_events = []
                else:
                    # self.buckets = [[None for _ in range(self.bucket_sizes[i])]
                    #                 for i in range(self.num_buckets)]
                    if not self.buckets:
                        self.buckets = [[None for _ in range(self.bucket_sizes[i])]
                                        for i in range(self.num_buckets)]
                    else:
                        assert len(self.buckets) == self.num_buckets, "len(buckets) = {}, expected {}".format(
                            len(self.buckets), self.num_buckets)
                        for b, bucket in enumerate(self.buckets):
                            assert len(bucket) == self.bucket_sizes[b], "len(buckets[{}]) = {}, expected {})".format(
                                b, len(buckets[b]), self.bucket_sizes[b])
                            for i in range(len(bucket)):
                                bucket[i] = None

                    if self.allreduce_communicators:
                        self.bucket_pgs = self.allreduce_communicators[0]
                        self.bucket_streams = self.allreduce_communicators[1]
                        self.bucket_events = [torch.cuda.Event(enable_timing=False,
                                            blocking=False) for _ in range(self.num_allreduce_streams)]
                    else:
                        if self.allreduce_different_streams:
                            if not self.bucket_pgs:
                                self.bucket_pgs = [dist.new_group() for _ in range(self.num_allreduce_streams)]
                                for i, bg in enumerate(self.bucket_pgs):
                                    print("rank {} created group {} with backend {}".format(
                                          dist.get_rank(), i, dist.get_backend(bg)))
                        if self.allreduce_different_streams:
                            if not self.bucket_streams:
                                self.bucket_streams = [torch.cuda.Stream() for _ in range(self.num_allreduce_streams)]
                                self.bucket_events = [torch.cuda.Event(enable_timing=False,
                                                      blocking=False) for _ in range(self.num_allreduce_streams)]
                        else:
                            if not self.bucket_streams:
                                self.bucket_streams = [torch.cuda.Stream()]
                                self.bucket_events = [torch.cuda.Event(enable_timing=False, blocking=False)]

                    self.buckets_ready_size = [0 for i in range(self.num_buckets)]
                    if(self.retain_allreduce_buffers):
                        self.allreduce_buffers = [None for _ in range(self.num_buckets)]
                    self.next_bucket = 0
                    self.ready_buckets_not_reduced = set()

                self.active_params = param_list

            self.callback_queued = False

        if self.prof:
            torch.cuda.nvtx.range_pop()

        return result
