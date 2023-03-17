import collections
import contextlib
import enum
import inspect
import io
import itertools
import threading
import types

import torch
from torch.distributed.distributed_c10d import _get_default_group
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
import distributed_adam_cuda

# Fallback to private functions if using PyTorch <1.13.0
try:
    from torch.distributed.distributed_c10d import get_global_rank
except ImportError:
    from torch.distributed.distributed_c10d import _get_global_rank
    get_global_rank = _get_global_rank
try:
    from torch.distributed.distributed_c10d import reduce_scatter_tensor
except ImportError:
    from torch.distributed.distributed_c10d import _reduce_scatter_base
    reduce_scatter_tensor = _reduce_scatter_base
try:
    from torch.distributed.distributed_c10d import all_gather_into_tensor
except ImportError:
    from torch.distributed.distributed_c10d import _all_gather_base
    all_gather_into_tensor = _all_gather_base

# Add args to coalescing manager if using PyTorch <=1.13.1
from torch.distributed.distributed_c10d import _coalescing_manager
if 'device' not in inspect.signature(_coalescing_manager).parameters.keys():
    _coalescing_manager_nodevice = _coalescing_manager
    @contextlib.contextmanager
    def _coalescing_manager(group, device, reqs):
        with _coalescing_manager_nodevice(group, reqs):
            yield

# Import optional CUDA kernels
_FOUND_DEPRECATED_FUSED_ADAM = False
try:
    import fused_adam_cuda
    _FOUND_DEPRECATED_FUSED_ADAM = True
except ImportError:
    import warnings
    warnings.warn(
        'Could not find recommended CUDA kernels when importing '
        '`DistributedFusedAdam`. '
        'For best performance, Apex should be installed with '
        '`--deprecated_fused_adam`.'
    )

def _ceildiv(numer, denom):
    """Assumes arguments are positive integers"""
    return (numer + denom - 1) // denom

def _round_to_multiple(number, multiple, round_up=True):
    """Assumes arguments are positive integers"""
    return (number+multiple-1 if round_up else number) // multiple * multiple

def _devices_match(device1, device2):
    """Whether two PyTorch devices are equivalent"""
    device1 = torch.device(device1)
    device2 = torch.device(device2)
    if device1.type != device2.type:
        return False
    if device1.type == 'cuda':
        index1 = device1.index
        index2 = device2.index
        if index1 is None:
            index1 = torch.cuda.current_device()
        if index2 is None:
            index2 = torch.cuda.current_device()
        if index1 != index2:
            return False
    return True

def _multi_tensor_copy(
        buffers_in,
        buffers_out,
        dummy_overflow_buf=None,
):
    """Copy between corresponding buffers

    Uses fused copy kernel if possible.
    """

    # Group buffers by device and dtype
    buffer_groups = collections.defaultdict(list)
    for buf_in, buf_out in zip(buffers_in, buffers_out):
        if buf_in.data_ptr() == buf_out.data_ptr() or buf_in.numel() == 0:
            # Nothing to be done if input and output buffers are same
            # or have no entries
            continue
        if buf_in.dtype == buf_out.dtype:
            # Just copy bytes if dtypes are same
            buf_in = buf_in.view(torch.uint8)
            buf_out = buf_out.view(torch.uint8)
        key = (buf_in.is_cuda, buf_in.dtype, buf_out.is_cuda, buf_out.dtype)
        buffer_groups[key].append((buf_in, buf_out))

    # Copy each group of buffers
    for key, buffers in buffer_groups.items():

        # Check if buffers support fused kernel
        is_cuda_in, dtype_in, is_cuda_out, dtype_out = key
        supported_dtypes = (torch.float32, torch.float16)
        use_fused_kernel = (
            (dtype_in in supported_dtypes and dtype_out in supported_dtypes)
            or
            (dtype_in == torch.uint8 and dtype_out == torch.uint8)
        )
        use_fused_kernel = use_fused_kernel and is_cuda_in and is_cuda_out

        # Copy buffers
        if use_fused_kernel and _FOUND_DEPRECATED_FUSED_ADAM:
            if dummy_overflow_buf is None:
                dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
            multi_tensor_applier(
                fused_adam_cuda.maybe_cast_mt,
                dummy_overflow_buf,
                list(zip(*buffers)),
            )
        else:
            for buf_in, buf_out in buffers:
                buf_out.copy_(buf_in)

@contextlib.contextmanager
def _disable_pre_forward_hook(param):
    """Prevent parameter from calling pre-forward hook"""
    hook_is_enabled = getattr(
        param,
        '_pre_forward_hook_is_enabled',
        False,
    )
    if hook_is_enabled:
        param._pre_forward_hook_is_enabled = False
    try:
        yield
    finally:
        if hook_is_enabled:
            param._pre_forward_hook_is_enabled = True

class DistributedFusedAdam(torch.optim.Optimizer):
    """Adam optimizer with ZeRO algorithm.

    Currently GPU-only. Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext --distributed_adam --deprecated_fused_adam``.

    This implements the ZeRO-2 algorithm, which distributes the
    optimizer state and gradients between parallel processes. In
    particular, the parameters are flattened, grouped into fixed-size
    buckets, and the optimizer state for each bucket is sharded over
    the parallel processes. Options are provided to overlap the
    gradient synchronization with the backward pass compute.

    Adam was proposed in `Adam: A Method for Stochastic
    Optimization`_, AdamW in `Decoupled Weight Decay Regularization`_,
    and ZeRO in `ZeRO: Memory Optimizations Toward Training Trillion
    Parameter Models`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        bias_correction (bool, optional): apply correction factor to
            moment estimates. (default: True)
        betas (Tuple[float, float], optional): coefficients used for
            computing running averages of gradient and its square.
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        adam_w_mode (boolean, optional): Decouple weight decay
            regularization (also known as AdamW algorithm) (default:
            True)
        weight_decay (float, optional): weight decay (L2 penalty)
            (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad
            variant of this algorithm from the paper
            `On the Convergence of Adam and Beyond`_ (default: False).
            This is not yet supported.
        dtype (torch.dtype, optional): datatype for optimizer state
            (default: torch.float32)
        grad_sync_dtype (torch.dtype, optional): datatype for gradient
            synchronization (default: same as dtype)
        param_sync_dtype (torch.dtype, optional): datatype for
            parameter synchronization (default: same as dtype)
        device (torch.device, optional): device for optimizer state
            (default: cuda). Currently only supports GPU with one GPU
            per process.
        process_group (torch.distributed.ProcessGroup, optional):
            parallel processes participating in optimizer (default:
            default group in torch.distributed). This group is
            interpreted as a 2D grid with dimensions
            distributed_size x redundant_size.
        distributed_process_group (torch.distributed.ProcessGroup,
            optional): parallel processes to distribute optimizer
            state over (default: same as process_group)
        redundant_process_group (torch.distributed.ProcessGroup,
            optional): parallel processes to replicate optimizer state
            over (default: group only containing calling process)
        average_grad_sync (bool, optional): whether to use average
            reduction for gradient synchronization rather than sum
            (default: True)
        overlap_grad_sync(boolean, optional): whether to overlap
            gradient synchronization with backward pass compute
            (default: True)
        overlap_param_sync(boolean, optional): whether to overlap
            parameter synchronization with forward pass compute
            (default: False). This is an experimental feature.
        bucket_cap_mb (float, optional): bucket size in megabytes
            (default: 100)
        pipeline_size (int, optional): number of buckets to process
            simultaneously in optimizer step (default: 2)
        contiguous_param_buffer (bool, optional): convert parameters
            into views into a large persistent buffer (default:
            False). This enables some performance optimizations (e.g.
            avoiding some memory copies), but may add memory overhead
            (e.g. if the memory allocator can't reuse the original
            parameter buffers).
        contiguous_grad_buffer (bool, optional): allocate gradient
            buckets out of a large persistent buffer (default: False).
            This allows individual parameter gradients to be accessed
            externally (see grad_buffer_view function). It enables
            some performance optimizations (e.g. avoiding some memory
            copies), but prevents some memory optimizations (e.g. the
            memory allocator can't reuse buffers for gradient
            buckets).
        store_params (bool, optional): store a distributed copy of the
            parameters as optimizer state (default: True). This may be
            desirable if the optimizer dtype has higher precision than
            the parameter dtype.
        store_param_remainders (bool, optional): if model is BF16 and
            optimizer is FP32, store bits required to reconstruct FP32
            params (default: False). This is an experimental feature.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
    .. _ZeRO\: Memory Optimizations Toward Training Trillion Parameter Models:
        https://arxiv.org/abs/1910.02054

    """

    class ParameterFragment:
        """Buffer ranges for a parameter fragment

        Describes corresponding regions in parameter buffer and
        parameter bucket.

        """
        def __init__(
                self,
                param_group_id,
                param_id,
                bucket_id,
                param_range,
                bucket_range,
                in_local_shard,
                shard_range,
                shard_bucket_range,
                shard_param_range,
        ):
            # Parameter group index
            self.param_group_id = param_group_id
            # Parameter index within parameter group
            self.param_id = param_id
            # Bucket index
            self.bucket_id = bucket_id
            # Range within flattened parameter buffer
            self.param_range = param_range
            # Range within bucket
            self.bucket_range = bucket_range
            # Whether fragment is in local shard of bucket
            self.in_local_shard = in_local_shard
            # Range within local shard
            self.shard_range = shard_range
            # Range of local fragment shard within bucket
            self.shard_bucket_range = shard_bucket_range
            # Range of local fragment shard within parameter
            self.shard_param_range = shard_param_range

    class StateBucket:
        """Optimizer state for a bucket"""
        def __init__(
                self,
                bucket_size,
                shard_size,
                dtype,
                device,
                contiguous_buffer_offset=0,
                store_params=False,
                store_param_remainders=False,
        ):
            # Size of parameter bucket
            self.bucket_size = bucket_size
            # Size of local shard of parameter bucket
            self.shard_size = shard_size
            # Size of the filled region in the bucket
            self.filled_size = 0
            # Offset to bucket in contiguous buffers
            self.contiguous_buffer_offset = contiguous_buffer_offset
            # Buffer ranges corresponding to parameter fragments
            self.fragments = []
            # Local shard of parameters
            self.params_shard = None
            if store_params:
                self.params_shard = torch.zeros(
                    [shard_size], dtype=dtype, device=device,
                )
            # Local shard of parameter remainders
            self.param_remainders_shard = None
            if store_param_remainders:
                self.param_remainders_shard = torch.zeros(
                    [shard_size], dtype=torch.int16, device=device,
                )
            # Local shard of first moment estimate
            self.exp_avg_shard = torch.zeros(
                [shard_size], dtype=dtype, device=device,
            )
            # Local shard of second moment estimate
            self.exp_avg_sq_shard = torch.zeros(
                [shard_size], dtype=dtype, device=device,
            )

    class GradientStatus(enum.Enum):
        """Status of gradients within a bucket"""
        # Gradients are ready to use
        READY = enum.auto()
        # Bucket is partially filled with unreduced gradients
        PARTIALLY_FILLED = enum.auto()
        # Bucket is fully filled with unreduced gradients
        FULLY_FILLED = enum.auto()
        # Asynchronous reduction is in progress
        SYNCING = enum.auto()

    class GradientBucket:
        """Gradient buffers and state for a bucket"""
        def __init__(self):
            # Local shard of gradients
            self.grads_shard = None
            # Local contribution to gradients
            self.grads_bucket = None
            # Buffer for gradient reduce-scatter
            self.sync_grads_shard = None
            # Status of gradients
            self.status = DistributedFusedAdam.GradientStatus.READY
            # Request object for asynchronous communication
            self.sync_request = None
            # Params that have generated grads
            self.grads_generated = set()

        def sync_wait(self):
            """Wait for asynchronous communication to finish"""
            if self.sync_request is not None:
                self.sync_request.wait()
            self.sync_request = None

    class ParameterStatus(enum.Enum):
        """Status of parameters within a bucket"""
        # Parameters are sharded between processes
        SHARDED = enum.auto()
        # Asynchronous communication is in progress
        SYNCING = enum.auto()
        # Parameters are ready to use
        READY = enum.auto()

    class ParameterBucket:
        """Parameter buffers and state for a bucket"""
        def __init__(self):
            # Local shard of parameters
            self.params_shard = None
            # Gathered parameter values
            self.params_bucket = None
            # Status of parameters
            self.status = DistributedFusedAdam.ParameterStatus.SHARDED
            # Request object for asynchronous communication
            self.sync_request = None
            # Params that have been updated
            self.params_updated = set()

        def sync_wait(self):
            """Wait for asynchronous communication to finish"""
            if self.sync_request is not None:
                self.sync_request.wait()
            self.sync_request = None

    # Enable custom logic for AMP grad scaling
    _step_supports_amp_scaling = True
    _custom_amp_unscale_grads = True

    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 adam_w_mode=True,
                 weight_decay=0.,
                 amsgrad=False,
                 dtype=torch.float32,
                 grad_sync_dtype=None,
                 param_sync_dtype=None,
                 device='cuda',
                 process_group=None,
                 distributed_process_group=None,
                 redundant_process_group=None,
                 average_grad_sync=True,
                 overlap_grad_sync=True,
                 overlap_param_sync=False,
                 bucket_cap_mb=100,
                 pipeline_size=2,
                 contiguous_param_buffer=False,
                 contiguous_grad_buffer=False,
                 store_params=True,
                 store_param_remainders=False,
    ):
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Adam options
        self.adam_w_mode = adam_w_mode
        if amsgrad:
            raise RuntimeError('DistributedFusedAdam does not support the AMSGrad variant.')

        # Datatype options
        if grad_sync_dtype is None:
            grad_sync_dtype = dtype
        if param_sync_dtype is None:
            param_sync_dtype = dtype
        supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if (dtype not in supported_dtypes
            or grad_sync_dtype not in supported_dtypes
            or param_sync_dtype not in supported_dtypes):
            raise RuntimeError(
                'Unsupported dtypes for DistributedFusedAdam '
                f'(dtype={dtype}, '
                f'grad_sync_dtype={grad_sync_dtype}, '
                f'param_sync_dtype={param_sync_dtype}))'
            )
        self.dtype = dtype
        self.grad_sync_dtype = grad_sync_dtype
        self.param_sync_dtype = param_sync_dtype

        # Device options
        if not _devices_match(device, 'cuda'):
            raise RuntimeError(
                'Invalid device for DistributedFusedAdam '
                f'(device={device})'
            )
        self.device = torch.device('cuda', torch.cuda.current_device())

        # Process groups
        self.process_group = (
            _get_default_group()
            if process_group is None
            else process_group
        )
        self.distributed_process_group = (
            self.process_group
            if distributed_process_group is None
            else distributed_process_group
        )
        self.redundant_process_group = redundant_process_group
        self.process_group_size = torch.distributed.get_world_size(self.process_group)
        self.distributed_rank = torch.distributed.get_rank(self.distributed_process_group)
        self.distributed_size = torch.distributed.get_world_size(self.distributed_process_group)
        self.redundant_size = (
            1
            if self.redundant_process_group is None
            else torch.distributed.get_world_size(self.redundant_process_group)
        )
        if self.process_group_size != self.distributed_size * self.redundant_size:
            raise RuntimeError(
                'Invalid process group configuration '
                f'(process group size = {self.process_group_size}, '
                f'distributed process group size = {self.distributed_size}, '
                f'redundant process group size = {self.redundant_size})'
            )
        self.process_group_root = get_global_rank(self.process_group, 0)

        # Use average reduction for grad sync
        self.average_grad_sync = average_grad_sync
        # Copy param grads to bucket as soon as available
        self.greedy_grad_copy = True
        # Synchronize grad buckets as soon as their grads are available
        self.overlap_grad_sync = overlap_grad_sync
        # Try synchronizing param buckets just before param is needed
        self.overlap_param_sync = overlap_param_sync
        # Number of buckets to synchronize at a time
        self.pipeline_size = pipeline_size

        # Store params or param remainders
        if store_param_remainders:
            if store_params:
                raise RuntimeError(
                    'Attempted to construct DistributedFusedAdam '
                    'with store_params=True and store_param_remainders=True'
                )
            if (self.dtype != torch.float32
                or self.param_sync_dtype != torch.bfloat16):
                raise RuntimeError(
                    'DistributedFusedAdam requires '
                    'BF16 params and FP32 optimizer state '
                    'when storing parameter remainders '
                    f'(dtype={self.dtype}, '
                    f'param_sync_dtype={self.param_sync_dtype}))'
                )
        self.store_params = store_params
        self.store_param_remainders = store_param_remainders

        # Determine bucket sizes
        dtype_size = torch.finfo(self.grad_sync_dtype).bits // 8
        self.alignment = 128 // dtype_size
        bucket_size = 1024*1024*bucket_cap_mb / dtype_size
        shard_size = int(bucket_size / self.distributed_size)
        shard_size = _round_to_multiple(shard_size, self.alignment, round_up=False)
        shard_size = max(shard_size, self.alignment)
        self.default_shard_size = shard_size

        # Optimizer state
        self.state['buckets'] = []
        self.state['step'] = 0

        # Gradient state
        self._grads_buckets = collections.defaultdict(self.GradientBucket)
        # Param state
        self._params_buckets = collections.OrderedDict()

        # Whether to allocate contiguous buffer for parameters
        self.contiguous_param_buffer = contiguous_param_buffer
        # Whether to allocate contiguous buffer for gradients
        self.contiguous_grad_buffer = contiguous_grad_buffer
        # Contiguous buffer for parameters
        self._param_buffer = None
        # Contiguous buffer for gradients
        self._grad_buffer = None

        # Whether to allocate contiguous buffer for gradients
        self.contiguous_grad_buffer = contiguous_grad_buffer
        # Contiguous buffer for gradients
        self._grad_buffer = None

        # Side streams for optimizer step and communication
        self._pipeline_streams = [torch.cuda.Stream() for _ in range(self.pipeline_size+1)]

        # Scale by factor before optimizer step. Used for grad
        # clipping and gradient scaler.
        self._grad_scale = torch.full([], 1.0, dtype=torch.float32, device=self.device)
        # Norm of parameter gradients. Used for gradient clipping and
        # gradient scaler.
        self._grad_norm = None

        # Dummy flag for multi-tensor kernels
        # Note: Apex multi-tensor kernels have a noop_flag argument
        # that is intended to detect non-finite values. It shouldn't
        # have any effect with the kernels used in the optimizer, but
        # we still set it to zero out of an abundance of caution.
        self._dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device=self.device)

        # Check if collectives have no_copy option
        self._gather_no_copy = (
            'no_copy' in inspect.getfullargspec(torch.distributed.gather).args
        )

        # Make sure parameter values are same across processes
        self._broadcast_params()

        # Lock for callbacks
        self._lock = threading.Lock()
        # Attach hooks for gradient synchronization
        self._register_post_backward_hooks()
        # Attach hooks for param synchronization
        if self.overlap_param_sync:
            self._register_pre_forward_hooks()

    def _broadcast_params(self):
        """Broadcast parameter values from root rank"""
        sync_requests = []
        process_group = self.process_group
        with _coalescing_manager(process_group, self.device, sync_requests):
            for param_group in self.param_groups:
                for param in param_group['params']:
                    sync_requests.append(
                        torch.distributed.broadcast(
                            param,
                            src=self.process_group_root,
                            group=process_group,
                            async_op=True,
                        )
                    )
        for req in sync_requests:
            req.wait()

    def _make_post_backward_hook(self, param, param_group_id, param_id):
        """Create callback function to call after param generates grad

        Lazily initialize parameter and try launching grad sync.

        """
        def post_backward_hook(*unused):
            if getattr(param, '_pre_forward_hook_is_enabled', False):
                raise RuntimeError(
                    'A parameter called its post-backward hook '
                    'before its pre-forward hook. '
                    'Please manually interact with the parameter '
                    'before the forward pass (e.g. by calling data_ptr) '
                    'or run DistributedFusedAdam with overlap_param_sync=False.'
                )
            with self._lock:
                need_to_initialize = 'fragments' not in self.state[param]
                if need_to_initialize:
                    self._init_param_state(param, param_group_id, param_id)
                if self.greedy_grad_copy:
                    self._grad_copy(param)
                    if self.overlap_grad_sync:
                        self._try_start_bucket_grad_sync(
                            params=[param],
                            ignore_last_bucket=need_to_initialize,
                        )
        return post_backward_hook

    def _register_post_backward_hooks(self):
        """Attach hooks for gradient synchronization"""
        self._grad_accs = []
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group['params']):
                if param.requires_grad:
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    hook = self._make_post_backward_hook(
                        param,
                        param_group_id,
                        param_id,
                    )
                    grad_acc.register_hook(hook)
                    self._grad_accs.append(grad_acc)

    def _make_pre_forward_hook(self, param, param_group_id, param_id):
        """Create callback function to call before param forward pass

        Make sure param has been synchronized and try launching next
        param sync.

        """
        def pre_forward_hook(*unused):
            with self._lock:
                if 'fragments' not in self.state[param]:
                    return
                self._param_copy(param)
                if self.overlap_param_sync:
                    self._try_start_bucket_param_sync()
        return pre_forward_hook

    def _register_pre_forward_hooks(self):
        """Attach hooks for parameter synchronization

        If _pre_forward_hook_is_enabled is set in a parameter, then
        the callback will be called the first time any of its
        attributes are accessed. This is hackily done by
        monkey-patching the parameter class, so proceed with caution.

        """
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group['params']):

                # Monkey-patch parameter class
                cls = param.__class__
                if not getattr(cls, '_has_pre_forward_hook', False):

                    # Monkey-patch magic methods to call __getattribute__
                    special_funcs = [
                        '__abs__', '__add__', '__and__',
                        '__bool__', '__complex__', '__contains__',
                        '__deepcopy__', '__delitem__', '__div__',
                        '__eq__', '__float__', '__floordiv__',
                        '__ge__', '__getitem__', '__gt__', '__iadd__',
                        '__iand__', '__idiv__', '__ifloordiv__',
                        '__ilshift__', '__imod__', '__imul__',
                        '__index__', '__int__', '__invert__',
                        '__ior__', '__ipow__', '__irshift__',
                        '__isub__', '__iter__', '__itruediv__',
                        '__ixor__', '__le__', '__len__', '__long__',
                        '__lshift__', '__lt__', '__matmul__',
                        '__mod__', '__mul__', '__neg__',
                        '__nonzero__', '__or__', '__pos__', '__pow__',
                        '__radd__', '__rand__', '__rdiv__',
                        '__reduce__', '__reduce_ex__', '__reversed__',
                        '__rfloordiv__', '__rlshift__', '__rmatmul__',
                        '__rmod__', '__rmul__', '__ror__', '__rpow__',
                        '__rrshift__', '__rshift__', '__rsub__',
                        '__rtruediv__', '__rxor__', '__setitem__',
                        '__sizeof__', '__sub__', '__torch_function__',
                        '__truediv__', '__xor__',
                    ]
                    for func_name in special_funcs:
                        def make_augmented_func():
                            base_func_name = f'_base_{func_name}'
                            def augmented_func(self, *args, **kwargs):
                                return getattr(self, base_func_name)(*args, **kwargs)
                            return augmented_func
                        setattr(cls, f'_base_{func_name}', getattr(cls, func_name))
                        setattr(cls, func_name, make_augmented_func())

                    # Monkey-patch __getattribute__ to call pre-forward hook
                    def make_getattribute():
                        special_attrs = {
                            '_pre_forward_hook_is_enabled',
                            '_pre_forward_hook',
                            '__del__', '__delattr__', '__dir__', '__getattr__',
                            '__getattribute__', '__hash__',
                            '__init__', '__new__', '__setattr__',
                        }
                        def getattribute_with_pre_forward_hook(self, name):
                            """Variant of __getattribute__ that can call pre-forward hook"""
                            if name not in special_attrs:
                                if getattr(self, '_pre_forward_hook_is_enabled', False):
                                    self._pre_forward_hook_is_enabled = False
                                    self._pre_forward_hook()
                            return object.__getattribute__(self, name)
                        return getattribute_with_pre_forward_hook
                    cls.__getattribute__ = make_getattribute()
                    cls._has_pre_forward_hook = True

                # Register pre-forward callback
                param._pre_forward_hook_is_enabled = False
                param._pre_forward_hook = self._make_pre_forward_hook(
                    param,
                    param_group_id,
                    param_id,
                )

    def init_param_buffer(self):
        """Allocate contiguous buffer for param buckets

        This converts the parameters into views into the contiguous
        buffer. This enables some performance optimizations (e.g.
        avoiding some memory copies), but may add memory overhead
        (e.g. if the memory allocator can't reuse the original
        parameter buffers). To minimize memory overhead, this buffer
        should be initialized before the first training step.

        """

        # Make sure all params are initialized
        self.contiguous_param_buffer = True
        self.init_params()

        # Construct param buffer
        if self.state['buckets']:
            buffer_size = max(
                bucket.contiguous_buffer_offset + bucket.bucket_size
                for bucket in self.state['buckets']
            )
        else:
            buffer_size = 0
        self._param_buffer = torch.zeros(
            [buffer_size],
            dtype=self.param_sync_dtype,
            device=self.device,
        )

        # Figure out corresponding positions in params and param buffer
        params = list(self.parameters())
        param_flat_views = []
        param_buffer_views = []
        for i, param in enumerate(params):
            fragment = self.state[param]['fragments'][0]
            bucket_id = fragment.bucket_id
            param_size = param.numel()
            bucket_start, _ = fragment.bucket_range
            buffer_offset = self.state['buckets'][bucket_id].contiguous_buffer_offset
            buffer_start = buffer_offset + bucket_start
            buffer_end = buffer_start + param_size
            buffer_view = self._param_buffer[buffer_start:buffer_end].detach()
            if not _devices_match(buffer_view.device, param.device):
                raise RuntimeError(
                    'Attempted to change a parameter with device={param.device} '
                    f'into a buffer view with device={view_buffer.device}'
                )
            if buffer_view.dtype != param.dtype:
                raise RuntimeError(
                    f'Attempted to change a parameter with dtype={param.dtype} '
                    f'into a buffer view with dtype={view_buffer.dtype}'
                )
            param_flat_views.append(param.detach().view(-1))
            param_buffer_views.append(buffer_view)

        # Copy values into param buffer
        _multi_tensor_copy(
            param_flat_views,
            param_buffer_views,
            dummy_overflow_buf=self._dummy_overflow_buf,
        )

        # Make all params a view into the param buffer
        for param, buffer_view in zip(params, param_buffer_views):
            param.data = buffer_view.view(param.size())

    def _init_grad_buffer(self):
        """Allocate contiguous buffer for grad buckets"""
        self.contiguous_grad_buffer = True
        self.init_params() # Make sure all params are initialized
        if self.state['buckets']:
            buffer_size = max(
                bucket.contiguous_buffer_offset + bucket.bucket_size
                for bucket in self.state['buckets']
            )
        else:
            buffer_size = 0
        self._grad_buffer = torch.zeros(
            [buffer_size],
            dtype=self.grad_sync_dtype,
            device=self.device,
        )

    def parameters(self):
        """Returns an iterator over optimizer parameters"""
        return itertools.chain.from_iterable(
            group['params'] for group in self.param_groups
        )

    def init_params(self, params=None):
        """Initialize optimizer state for parameters

        Ignores parameters that have already been initialized.

        Arguments:
            params (iterable, optional): parameters to initialize
                (default: all parameters)

        """

        # Default cases
        if params is None:
            params = self.parameters()
        elif isinstance(params, torch.Tensor):
            params = [params]

        # Ignore parameters that have already been initialized
        params = [
            param
            for param in params
            if 'fragments' not in self.state[param]
        ]
        if not params:
            return

        # Get indices corresponding to parameters
        id_map = dict()
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group['params']):
                id_map[param] = (param_group_id, param_id)

        # Initialize parameters
        for param in params:
            if param in id_map:
                param_group_id, param_id = id_map[param]
                self._init_param_state(param, param_group_id, param_id)

    def init_params_bucket(self, params):
        """Initialize optimizer state for parameters in one effective bucket

        The buckets corresponding to the provided parameters are
        configured so they all perform communication together. Ignores
        parameters that have already been initialized.

        Arguments:
            params (iterable): parameters to initialize

        """

        # Ignore parameters that have already been initialized
        if isinstance(params, torch.Tensor):
            params = [params]
        params = [
            param
            for param in params
            if 'fragments' not in self.state[param]
        ]
        if not params:
            return

        # Get indices corresponding to parameters
        id_map = dict()
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group['params']):
                id_map[param] = [param_group_id, param_id]
        param_ids = [tuple([param] + id_map[param]) for param in params]

        # Mark existings bucket as fully filled
        for bucket in self.state['buckets']:
            bucket.filled_size = bucket.bucket_size

        # Initialize optimizer state for parameters
        start_bucket_id = len(self.state['buckets'])
        self.init_params(params)
        end_bucket_id = len(self.state['buckets'])

        # Make sure all added buckets depend on provided params
        for bucket_id in range(start_bucket_id, end_bucket_id):
            bucket = self.state['buckets'][bucket_id]
            bucket_size = bucket.bucket_size
            bucket.filled_size = bucket_size
            ids_in_bucket = set(
                (fragment.param_group_id, fragment.param_id)
                for fragment in bucket.fragments
            )
            for param, param_group_id, param_id in param_ids:
                if (param_group_id, param_id) not in ids_in_bucket:
                    param_size = param.numel()
                    fragment = self.ParameterFragment(
                        param_group_id=param_group_id,
                        param_id=param_id,
                        bucket_id=bucket_id,
                        param_range=(param_size, param_size),
                        bucket_range=(bucket_size, bucket_size),
                        in_local_shard=False,
                        shard_range=(None, None),
                        shard_bucket_range=(None, None),
                        shard_param_range=(None, None),
                    )
                    self.state[param]['fragments'].append(fragment)
                    bucket.fragments.append(fragment)

    def _init_param_state(
            self,
            param,
            param_group_id,
            param_id,
    ):
        """Initialize optimizer state for a parameter"""

        # Return immediately if already initialized
        if 'fragments' in self.state[param]:
            return
        self.state[param]['fragments'] = []

        # Make sure there is at least one bucket
        if not self.state['buckets']:
            shard_size = self.default_shard_size
            bucket_size = shard_size * self.distributed_size
            buffer_offset = 0
            self.state['buckets'].append(
                self.StateBucket(
                    bucket_size,
                    shard_size,
                    self.dtype,
                    self.device,
                    contiguous_buffer_offset=buffer_offset,
                    store_params=self.store_params,
                    store_param_remainders=self.store_param_remainders,
                )
            )

        # Split parameter values into fragments
        # Note: Each fragment resides within a bucket
        param_start = 0
        param_size = param.numel()
        while param_start < param_size:

            # Get current bucket
            bucket_id = len(self.state['buckets']) - 1
            bucket = self.state['buckets'][bucket_id]
            fragment_id = len(bucket.fragments)
            bucket_size = bucket.bucket_size
            shard_size = bucket.shard_size

            # Determine fragment position within bucket
            bucket_start = _round_to_multiple(
                bucket.filled_size,
                self.alignment,
                round_up=True,
            )
            fragment_size = min(param_size-param_start, bucket_size-bucket_start)
            param_end = param_start + fragment_size
            bucket_end = bucket_start + fragment_size

            # Create new bucket if current one is full
            if fragment_size <= 0:
                shard_size = self.default_shard_size
                bucket_size = shard_size * self.distributed_size
                buffer_offset = bucket.contiguous_buffer_offset + bucket.bucket_size
                self.state['buckets'].append(
                    self.StateBucket(
                        bucket_size,
                        shard_size,
                        self.dtype,
                        self.device,
                        contiguous_buffer_offset=buffer_offset,
                        store_params=self.store_params,
                        store_param_remainders=self.store_param_remainders,
                    )
                )
                continue

            # Fragment position within local shard
            shard_id = self.distributed_rank
            shard_start = bucket_start - shard_size*shard_id
            shard_end = bucket_end - shard_size*shard_id
            shard_start = min(max(shard_start, 0), shard_size)
            shard_end = min(max(shard_end, 0), shard_size)
            in_local_shard = shard_start < shard_end
            if in_local_shard:
                shard_bucket_start = shard_start + shard_size*shard_id
                shard_bucket_end = shard_bucket_start + shard_end - shard_start
                shard_param_start = shard_bucket_start - bucket_start + param_start
                shard_param_end = shard_param_start + shard_end - shard_start
            else:
                shard_start, shard_end = None, None
                shard_bucket_start, shard_bucket_end = None, None
                shard_param_start, shard_param_end = None, None

            # Record fragment info
            fragment = self.ParameterFragment(
                param_group_id=param_group_id,
                param_id=param_id,
                bucket_id=bucket_id,
                param_range=(param_start,param_end),
                bucket_range=(bucket_start,bucket_end),
                in_local_shard=in_local_shard,
                shard_range=(shard_start,shard_end),
                shard_bucket_range=(shard_bucket_start,shard_bucket_end),
                shard_param_range=(shard_param_start,shard_param_end),
            )
            self.state[param]['fragments'].append(fragment)
            bucket.fragments.append(fragment)
            bucket.filled_size = bucket_end
            param_start = param_end

        # Initialize main param buffer
        if self.store_params:
            for fragment in self.state[param]['fragments']:
                if fragment.in_local_shard:
                    bucket = self.state['buckets'][fragment.bucket_id]
                    param_start, param_end = fragment.shard_param_range
                    shard_start, shard_end = fragment.shard_range
                    model_param_fragment = param.detach().view(-1)[param_start:param_end]
                    main_param_fragment = bucket.params_shard[shard_start:shard_end]
                    main_param_fragment.copy_(model_param_fragment)

    def zero_grad(self, set_to_none=False):
        """Clear parameter gradients"""

        # Reset bucket buffers
        self._grads_buckets.clear()

        # Construct views into contiguous grad buffer, if needed
        if self.contiguous_grad_buffer:
            if self._grad_buffer is None:
                self._init_grad_buffer()
            self._grad_buffer.zero_()
            for bucket_id, bucket in enumerate(self.state['buckets']):
                bucket_size = bucket.bucket_size
                buffer_start = bucket.contiguous_buffer_offset
                buffer_end = buffer_start + bucket_size
                grad_buffer = self._grad_buffer[buffer_start:buffer_end]
                self._grads_buckets[bucket_id].grads_bucket = grad_buffer

        # Reset param grads
        for param in self.parameters():
            with _disable_pre_forward_hook(param):
                if set_to_none:
                    param.grad = None
                elif (self.contiguous_grad_buffer
                      and param.dtype == self.grad_sync_dtype
                      and _devices_match(param.device, self.device)):
                    param.grad = self.grad_buffer_view(param)
                elif param.grad is not None:
                    param.grad.zero_()

        # Reset other state
        self._grad_scale = torch.full([], 1.0, dtype=torch.float32, device=self.device)
        self._grad_norm = None
        self._dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device=self.device)

    def _grad_copy(self, param):
        """Copy parameter gradients to buckets"""

        # Initialize parameter if needed
        if 'fragments' not in self.state[param]:
            for param_group_id, group in enumerate(self.param_groups):
                for param_id, param_ in enumerate(group['params']):
                    if param == param_:
                        self._init_param_state(param, param_group_id, param_id)
            if 'fragments' not in self.state[param]:
                raise RuntimeError(
                    'Could not initialize DistributedFusedAdam with parameter'
                )

        # Copy param grad to buckets
        for fragment in self.state[param]['fragments']:

            # Get fragment position
            bucket_id = fragment.bucket_id
            bucket = self._grads_buckets[bucket_id]
            bucket_size = self.state['buckets'][bucket_id].bucket_size
            grad_start, grad_end = fragment.param_range
            bucket_start, bucket_end = fragment.bucket_range

            # Set reduction status
            if bucket.status == self.GradientStatus.SYNCING:
                self._finish_bucket_grad_sync()
            bucket.status = self.GradientStatus.PARTIALLY_FILLED

            # Allocate gradient buffer if needed
            if bucket.grads_bucket is None and self.contiguous_grad_buffer:
                if self._grad_buffer is None:
                    self._init_grad_buffer()
                buffer_start = self.state['buckets'][bucket_id].contiguous_buffer_offset
                buffer_end = buffer_start + bucket_size
                grad_buffer = self._grad_buffer[buffer_start:buffer_end]
                if (bucket.grads_shard is None
                    or bucket.grads_shard.data_ptr() != grad_buffer.data_ptr()):
                    bucket.grads_bucket = grad_buffer
                    bucket.grads_bucket.zero_()
            if bucket.grads_bucket is None:
                bucket.grads_bucket = torch.zeros(
                    [bucket_size],
                    dtype=self.grad_sync_dtype,
                    device=self.device,
                )

            # Copy param grad to bucket
            if param.grad is not None:
                grad_in = param.grad.detach().view(-1)[grad_start:grad_end]
                grad_out = bucket.grads_bucket[bucket_start:bucket_end]
                if grad_in.data_ptr() != grad_out.data_ptr():
                    grad_out.add_(grad_in)

        # Free param grad buffer
        param.grad = None

    def _param_copy(self, params):
        """Update parameters with values from parameter buckets"""

        # Get parameter fragments to be synchronized
        if isinstance(params, torch.Tensor):
            params = [params]
        fragments = []
        for param in params:
            if 'fragments' in self.state[param]:
                fragments.extend(
                    fragment
                    for fragment in self.state[param]['fragments']
                    if fragment.bucket_id in self._params_buckets
                )

        # Make sure all needed buckets have been synchronized
        buckets = collections.OrderedDict()
        for fragment in fragments:
            bucket_id = fragment.bucket_id
            bucket = self._params_buckets[bucket_id]
            buckets[bucket] = bucket.status
        if any(status != self.ParameterStatus.READY
               for bucket, status in buckets.items()):
            self._start_bucket_param_sync(buckets.keys())
            self._finish_bucket_param_sync()

        # Copy values from bucket buffers to params
        params_in = []
        params_out = []
        for fragment in fragments:
            bucket_id = fragment.bucket_id
            param_group_id = fragment.param_group_id
            param_id = fragment.param_id
            bucket_start, bucket_end = fragment.bucket_range
            param_start, param_end = fragment.param_range
            if param_end > param_start:
                bucket = self._params_buckets[bucket_id]
                param = self.param_groups[param_group_id]['params'][param_id]
                params_in.append(bucket.params_bucket[bucket_start:bucket_end])
                params_out.append(param.detach().view(-1)[param_start:param_end])
        _multi_tensor_copy(
            params_in,
            params_out,
            dummy_overflow_buf=self._dummy_overflow_buf,
        )

        # Delete buckets if possible
        for fragment in fragments:
            bucket_id = fragment.bucket_id
            bucket = self._params_buckets[bucket_id]
            bucket_fragments = self.state['buckets'][bucket_id].fragments
            param_group_id = fragment.param_group_id
            param_id = fragment.param_id
            param = self.param_groups[param_group_id]['params'][param_id]
            bucket.params_updated.add(param)
            if len(bucket.params_updated) == len(bucket_fragments):
                del self._params_buckets[bucket_id]

    def grad_buffer_view(self, param):
        """Construct view into grad buffer corresponding to param

        Assumes optimizer is using a contiguous grad buffer.

        """

        # Initialize contiguous grad buffer if needed
        assert self.contiguous_grad_buffer
        if self._grad_buffer is None:
            self._init_grad_buffer()

        # Figure out corresponding position in grad buffer
        fragment = self.state[param]['fragments'][0]
        bucket_id = fragment.bucket_id
        param_size = param.numel()
        bucket_start, _ = fragment.bucket_range
        buffer_offset = self.state['buckets'][bucket_id].contiguous_buffer_offset
        buffer_start = buffer_offset + bucket_start
        buffer_end = buffer_start + param_size

        # Construct view into grad buffer
        flat_buffer = self._grad_buffer[buffer_start:buffer_end]
        return flat_buffer.detach().view(param.size())

    def _force_bucket_grad_sync(self):
        """Ensure that all gradient buckets are synchronized"""

        # Synchronize all unsynchronized buckets
        Status = self.GradientStatus
        buckets = []
        for bucket_id, bucket in sorted(self._grads_buckets.items()):
            if bucket.status not in (Status.READY, Status.SYNCING):
                buckets.append(bucket)
                if bucket.grads_bucket is None:
                    bucket_size = self.state['buckets'][bucket_id].bucket_size
                    bucket.grads_bucket = torch.zeros(
                        [bucket_size],
                        dtype=self.grad_sync_dtype,
                        device=self.device,
                    )
        if buckets:
            self._start_bucket_grad_sync(buckets)
        self._finish_bucket_grad_sync()

        # Fill any unsynchronized gradients with zeros
        for bucket_id in range(len(self.state['buckets'])):
            bucket = self._grads_buckets[bucket_id]
            if bucket.grads_shard is None:
                shard_size = self.state['buckets'][bucket_id].shard_size
                bucket.grads_shard = torch.zeros(
                    [shard_size],
                    dtype=self.grad_sync_dtype,
                    device=self.device,
                )

    def _try_start_bucket_grad_sync(
            self,
            params=[],
            ignore_last_bucket=False,
    ):
        """Attempt to launch gradient synchronization

        Launches gradient synchronization if any bucket has receieved
        all its expected gradients. Gradient synchronization is
        asynchronous.

        Arguments:
            params (iterable): parameters that have had their
                gradients copied to buckets
            ignore_last_bucket (bool): avoid synchronizing last bucket
                until all gradients have been generated. This avoids
                excessive synchronization when initializing buckets in
                the first backward pass.

        """

        # Register params that have generated grads
        for param in params:
            for fragment in self.state[param]['fragments']:
                bucket_id = fragment.bucket_id
                bucket = self._grads_buckets[bucket_id]
                bucket_fragments = self.state['buckets'][bucket_id].fragments
                bucket.grads_generated.add(param)
                if len(bucket.grads_generated) == len(bucket_fragments):
                    bucket.status = self.GradientStatus.FULLY_FILLED
                    if bucket.grads_bucket is None:
                        bucket_size = self.state['buckets'][bucket_id].bucket_size
                        bucket.grads_bucket = torch.zeros(
                            [bucket_size],
                            dtype=self.grad_sync_dtype,
                            device=self.device,
                        )

        # Launch reductions if enough buckets are ready
        filled_buckets = []
        for bucket_id, bucket in sorted(self._grads_buckets.items()):
            if ignore_last_bucket and bucket_id == len(self.state['buckets'])-1:
                continue
            if bucket.status == self.GradientStatus.FULLY_FILLED:
                filled_buckets.append(bucket)
        if filled_buckets:
            self._start_bucket_grad_sync(filled_buckets)

    def _start_bucket_grad_sync(self, buckets):
        """Synchronize gradient buckets

        Gradient synchronization is asynchronous. Involves
        reduce-scatter over distributed process group and allreduce
        over redundant process group. Assumes grad bucket buffers are
        already initialized.

        """

        # Complete any outstanding grad syncs
        # Note: Not needed with contiguous grad buffer since there is
        # no memory benefit from eagerly freeing grad buffers.
        if not self.contiguous_grad_buffer:
            self._finish_bucket_grad_sync()

        # Reduction operation
        if self.average_grad_sync:
            reduce_op = torch.distributed.ReduceOp.AVG
        else:
            reduce_op = torch.distributed.ReduceOp.SUM

        # Initialize grad state and buffers
        for bucket in buckets:
            if bucket.status == self.GradientStatus.SYNCING:
                self._finish_bucket_grad_sync()
            bucket.status = self.GradientStatus.SYNCING
            bucket.grads_generated.clear()
            if self.distributed_size == 1:
                bucket.sync_grads_shard = bucket.grads_bucket
            else:
                bucket_size = bucket.grads_bucket.numel()
                shard_size = bucket_size // self.distributed_size
                bucket.sync_grads_shard = torch.empty(
                    [shard_size],
                    dtype=self.grad_sync_dtype,
                    device=self.device,
                )

        # Side stream for communication
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        comm_stream.wait_stream(main_stream)

        # Reduce-scatter over distributed process group
        if self.distributed_size > 1:
            with torch.cuda.stream(comm_stream):
                for bucket in buckets:
                    bucket.sync_wait()
                sync_requests = []
                group = self.distributed_process_group
                with _coalescing_manager(group, self.device, sync_requests):
                    for bucket in buckets:
                        bucket.sync_request = (
                            reduce_scatter_tensor(
                                bucket.sync_grads_shard,
                                bucket.grads_bucket,
                                op=reduce_op,
                                group=group,
                                async_op=True,
                            )
                        )
                        sync_requests.append(bucket.sync_request)

        # All-reduce over redundant process group
        if self.redundant_size > 1:
            with torch.cuda.stream(comm_stream):
                for bucket in buckets:
                    bucket.sync_wait()
                sync_requests = []
                group = self.redundant_process_group
                with _coalescing_manager(group, self.device, sync_requests):
                    for bucket in buckets:
                        bucket.sync_request = (
                            torch.distributed.all_reduce(
                                bucket.sync_grads_shard,
                                op=reduce_op,
                                group=group,
                                async_op=True,
                            )
                        )
                        sync_requests.append(bucket.sync_request)

    def _finish_bucket_grad_sync(self):
        """Wait for any gradient synchronizations that are in progress"""
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        main_stream.wait_stream(comm_stream)
        for bucket_id, bucket in sorted(self._grads_buckets.items()):
            if bucket.status == self.GradientStatus.SYNCING:

                # Finish asynchronous communication
                bucket.sync_wait()

                # Accumulate gradient in local shard
                if bucket.grads_shard is None:
                    bucket.grads_shard = bucket.sync_grads_shard
                else:
                    bucket.grads_shard.add_(bucket.sync_grads_shard)
                bucket.grads_bucket = None
                bucket.sync_grads_shard = None

                # Reset status
                bucket.status = self.GradientStatus.READY

                # Cached gradient norm has been invalidated
                self._grad_norm = None

    def _try_start_bucket_param_sync(
            self,
            params=None,
    ):
        """Attempt to launch parameter synchronization

        Launches parameter synchronization for buckets corresponding
        to provided parameters, if needed. If parameters are not
        provided and no other synchronizations are in progress,
        attempts to find a parameter that still requires
        synchronization. Parameter synchronization is asynchronous.

        Arguments:
            params (iterable, optional): parameters to synchronize

        """

        # Default behavior: only launch param sync if no other syncs
        # are in progress
        if params is None:
            params = []
            if any(bucket.status == self.ParameterStatus.SYNCING
                   for bucket in self._params_buckets.values()):
                return
            for bucket_id, bucket in self._params_buckets.items():
                if bucket.status == self.ParameterStatus.SHARDED:
                    fragment = self.state['buckets'][bucket_id].fragments[-1]
                    param_group_id = fragment.param_group_id
                    param_id = fragment.param_id
                    param = self.param_groups[param_group_id]['params'][param_id]
                    params.append(param)
                    break

        # Find buckets corresponding to params
        bucket_ids = set()
        for param in params:
            bucket_ids.update(
                fragment.bucket_id
                for fragment in self.state[param]['fragments']
            )
        buckets = [
            self._params_buckets[bucket_id]
            for bucket_id in sorted(bucket_ids)
            if bucket_id in self._params_buckets
        ]
        buckets = [
            bucket
            for bucket in buckets
            if bucket.status == self.ParameterStatus.SHARDED
        ]

        # Launch param sync if needed
        if buckets:
            self._start_bucket_param_sync(buckets)

    def _start_bucket_param_sync(self, buckets):
        """Synchronize parameter buckets

        Parameter synchronization is asynchronous. Involves all-gather
        over distributed process group. Assumes param shard buffers
        are already initialized.

        """

        # Complete any outstanding param syncs
        self._finish_bucket_param_sync()

        # Initialize param state and buffers
        buckets = [
            bucket
            for bucket in buckets
            if bucket.status == self.ParameterStatus.SHARDED
        ]
        for bucket in buckets:
            bucket.status = self.ParameterStatus.SYNCING
            if self.distributed_size == 1:
                bucket.params_bucket = bucket.params_shard
            elif bucket.params_bucket is None:
                shard_size = bucket.params_shard.numel()
                bucket_size = shard_size * self.distributed_size
                bucket.params_bucket = torch.empty(
                    [bucket_size],
                    dtype=self.param_sync_dtype,
                    device=self.device,
                )

        # Side stream for communication
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        comm_stream.wait_stream(main_stream)

        # All-gather over distributed process group
        if self.distributed_size > 1:
            with torch.cuda.stream(comm_stream):
                for bucket in buckets:
                    bucket.sync_wait()
                sync_requests = []
                group = self.distributed_process_group
                with _coalescing_manager(group, self.device, sync_requests):
                    for bucket in buckets:
                        bucket.sync_request = (
                            all_gather_into_tensor(
                                bucket.params_bucket,
                                bucket.params_shard,
                                group=group,
                                async_op=True,
                            )
                        )
                        sync_requests.append(bucket.sync_request)

    def _finish_bucket_param_sync(self):
        """Wait for any param synchronizations that are in progress"""
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        main_stream.wait_stream(comm_stream)
        for bucket_id, bucket in self._params_buckets.items():
            if bucket.status == self.ParameterStatus.SYNCING:
                bucket.sync_wait()
                bucket.params_shard = None
                bucket.status = self.ParameterStatus.READY

    @contextlib.contextmanager
    def no_sync(self, greedy_grad_copy=False):
        """Disable overlapped gradient synchronization

        Context manager that is similar to
        torch.nn.parallel.DistributedDataParallel.no_sync. The
        gradients can be synchronized by calling grad_sync or step. If
        overlapped gradient synchronization is enabled, gradients can
        also be synchronized by leaving the context and performing a
        backward pass.

        Arguments:
            greedy_grad_copy (bool, optional): copy parameter
                gradients to buckets as soon as they are generated
                (default: False)

        """
        old_greedy_grad_copy = self.greedy_grad_copy
        old_overlap_grad_sync = self.overlap_grad_sync
        self.greedy_grad_copy = greedy_grad_copy
        self.overlap_grad_sync = False
        try:
            yield
        finally:
            self.greedy_grad_copy = old_greedy_grad_copy
            self.overlap_grad_sync = old_overlap_grad_sync

    def grad_sync(self):
        """Ensure that all gradients are synchronized"""
        for bucket in self.state['buckets']:
            for fragment in bucket.fragments:
                param_group_id = fragment.param_group_id
                param_id = fragment.param_id
                param = self.param_groups[param_group_id]['params'][param_id]
                if param.grad is not None:
                    self._grad_copy(param)
                    if not self.contiguous_grad_buffer:
                        self._try_start_bucket_grad_sync(
                            params=[param],
                            ignore_last_bucket=False,
                        )
        self._force_bucket_grad_sync()

    def param_sync(self):
        """Ensure that all parameters are synchronized"""
        if self.contiguous_param_buffer:
            self._param_copy(self.parameters())
        else:
            while self._params_buckets:
                bucket_id, bucket = next(iter((self._params_buckets.items())))
                for fragment in reversed(self.state['buckets'][bucket_id].fragments):
                    param_id = fragment.param_id
                    param_group_id = fragment.param_group_id
                    param = self.param_groups[param_group_id]['params'][param_id]
                    self._param_copy(param)
        self._params_buckets.clear()

    def _local_grad_norm(self, parameters=None, norm_type=2.0):
        """Local contribution to parameter gradient norm

        Returns square of 2-norm. Other norms are not yet supported.

        If no parameters are provided, the norm is computed for all
        parameters in optimizer. Provided parameters are assumed to be
        in optimizer and to require gradients.

        """
        norm_type = float(norm_type)
        assert norm_type == 2.0

        # Make sure that gradients have been reduced
        self.grad_sync()

        # Check if provided parameters are subset of all parameters
        if parameters is not None:
            params_set = set(parameters)
            all_params_set = set()
            for bucket in self.state['buckets']:
                for fragment in bucket.fragments:
                    param_group_id = fragment.param_group_id
                    param_id = fragment.param_id
                    all_params_set.add(
                        self.param_groups[param_group_id]['params'][param_id]
                    )
            if not params_set.issubset(all_params_set):
                raise RuntimeError(
                    'Attempted to compute gradient norm for a parameter '
                    'that is not managed by DistributedFusedAdam'
                )
            if params_set == all_params_set:
                parameters = None

        if parameters is None:
            # Compute norm of all local gradients
            grad_norm_sq = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                self._dummy_overflow_buf,
                [[bucket.grads_shard for bucket in self._grads_buckets.values()]],
                False,
            )[0] ** 2
        else:
            # Compute norm of selected local gradients
            grads = []
            for param in parameters:
                if 'fragments' not in self.state[param]:
                    continue
                for fragment in self.state[param]['fragments']:
                    if fragment.in_local_shard:
                        bucket = self._grads_buckets[fragment.bucket_id]
                        shard_start, shard_end = fragment.shard_range
                        if shard_end > shard_start:
                            grads.append(bucket.grads_shard[shard_start:shard_end])
            if grads:
                grad_norm_sq = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    self._dummy_overflow_buf,
                    [grads],
                    False,
                )[0] ** 2
            else:
                grad_norm_sq = torch.zeros([1], dtype=self.dtype, device=self.device)

        grad_norm_sq = grad_norm_sq.detach()
        grad_norm_sq = grad_norm_sq.to(dtype=self.dtype, device=self.device)
        grad_norm_sq = grad_norm_sq.view([])
        return grad_norm_sq

    def grad_norm(self, parameters=None, norm_type=2.0, force=False):
        """Gradient norm of parameters in optimizer

        The norm is computed over all gradients together, as if they
        were concatenated into a single vector. All provided
        parameters must be managed by optimizer.

        The computed value is cached to avoid redundant communication.

        Arguments:
            parameters (iterable, optional): an iterable of parameters
                in optimizer (default: all parameters in optimizer).
            norm_type (float or int, optional): type of the used
                p-norm (default: 2). Only 2-norm is currently
                supported.
            force (bool, optional): ignore cached value and force norm
                computation (default: False).

        """
        if force or self._grad_norm is None:
            norm_type = float(norm_type)
            assert norm_type == 2.0
            grad_norm_sq = self._local_grad_norm(
                parameters=parameters,
                norm_type=norm_type,
            )
            torch.distributed.all_reduce(
                grad_norm_sq,
                op=torch.distributed.ReduceOp.SUM,
                group=self.distributed_process_group,
            )
            self._grad_norm = grad_norm_sq.sqrt()
        grad_norm = self._grad_norm * self._grad_scale
        return grad_norm.detach()

    def clip_grad_norm(self, max_norm, parameters=None, norm_type=2.0):
        """Clips gradient norm of parameters in optimizer

        The norm is computed over all gradients together, as if they
        were concatenated into a single vector. The scaling is
        deferred until the optimizer step, which should be called
        immediately after this function.

        The computed grad norm is cached to avoid redundant
        communication.

        Arguments:
            max_norm (float or int): max norm of the gradients
            parameters (iterable, optional): an iterable of parameters
                in optimizer (default: all parameters in optimizer).
            norm_type (float or int, optional): type of the used
                p-norm (default: 2)

        """
        assert max_norm > 0
        total_norm = self.grad_norm(parameters=parameters, norm_type=norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        self._grad_scale *= clip_coef_clamped
        return total_norm

    def unscale_grads(self, inv_scale, *args):
        """Custom unscale function for use by AMP gradient scaler

        Overflow checking is deferred to optimization step.

        Arguments:
            inv_scale (torch.Tensor): factor to multiply gradients

        """
        self._grad_scale *= inv_scale.view([])
        return { self.device: torch.zeros(1, dtype=torch.float32, device=self.device) }

    def step(self, closure=None, *, grad_scaler=None):
        """Apply Adam optimizer step

        Arguments:
            closure (callable, optional): closure to recompute loss
                (default: None)
            grad_scaler (torch.cuda.amp.GradScaler, optional):
                gradient scaler (default: None)

        """

        # Apply closure
        loss = None
        if closure is not None:
            loss = closure()

        # Make sure that parameters and gradients are synchronized
        self.param_sync()
        self.grad_sync()

        # Apply gradient scaler if provided
        # Note: We compute gradient norm to check for non-finite
        # values. This is more conservative and compute intensive than
        # directly checking, but it avoids extra communication if we
        # have already computed gradient norm e.g. for gradient
        # clipping.
        if grad_scaler is not None:
            grad_scaler_state = grad_scaler._per_optimizer_states[id(self)]
            GradScalerOptState = torch.cuda.amp.grad_scaler.OptState
            if grad_scaler_state['stage'] is GradScalerOptState.READY:
                assert grad_scaler._scale is not None
                self._grad_scale /= grad_scaler._scale.view([])
            grad_norm = self.grad_norm()
            found_inf = torch.logical_not(torch.isfinite(grad_norm))
            scaler_state = grad_scaler._per_optimizer_states[id(self)]
            scaler_state['found_inf_per_device'] = {found_inf.device: found_inf.float()}
            if found_inf.item():
                return
        self._grad_scale = self._grad_scale.to(dtype=torch.float32, device=self.device)

        # Initialize param shard buffers
        for bucket_id in reversed(range(len(self.state['buckets']))):
            bucket = self.ParameterBucket()
            self._params_buckets[bucket_id] = bucket
            shard_size = self.state['buckets'][bucket_id].shard_size
            if self.contiguous_param_buffer:
                if self._param_buffer is None:
                    self.init_param_buffer()
                bucket_size = self.state['buckets'][bucket_id].bucket_size
                buffer_start = self.state['buckets'][bucket_id].contiguous_buffer_offset
                buffer_end = buffer_start + bucket_size
                bucket.params_bucket = (
                    self._param_buffer[buffer_start:buffer_end]
                )
                bucket_start = self.distributed_rank * shard_size
                bucket_end = bucket_start + shard_size
                bucket.params_shard = (
                    bucket.params_bucket[bucket_start:bucket_end]
                )
            else:
                bucket.params_shard = torch.empty(
                    [shard_size],
                    dtype=self.param_sync_dtype,
                    device=self.device,
                )

        # Apply optimizer step and synchronize params
        self.state['step'] += 1
        if self.distributed_size > 1 and self.overlap_param_sync and self.state['buckets']:
            # Local step and non-blocking param sync
            # Note: Overlap param sync of first buckets with optimizer
            # step of remaining buckets.

            # Get buckets containing "first" parameter
            fragment = self.state['buckets'][-1].fragments[-1]
            param_group_id = fragment.param_group_id
            param_id = fragment.param_id
            param = self.param_groups[param_group_id]['params'][param_id]
            first_bucket_ids = sorted(
                fragment.bucket_id
                for fragment in self.state[param]['fragments']
            )

            # Local step and launch param sync for first buckets
            self._local_step(first_bucket_ids)
            self._start_bucket_param_sync(
                self._params_buckets[bucket_id]
                for bucket_id in first_bucket_ids
            )

            # Local step for remaining buckets
            first_bucket_ids = set(first_bucket_ids)
            self._local_step(
                bucket_id
                for bucket_id in range(len(self.state['buckets']))
                if bucket_id not in first_bucket_ids
            )

            # Enable pre-forward hook
            for param in self.parameters():
                param._pre_forward_hook_is_enabled = True

        else:
            # Local step and blocking param sync
            self._local_step(list(range(len(self.state['buckets']))))
            self.param_sync()

        return loss

    def _local_step(self, bucket_ids):
        """Apply optimizer step to local shard of parameter buckets

        Arguments:
            bucket_ids (iterable): bucket indices

        """

        # Optimized implementation with BF16 params and 16-bit param
        # remainders
        if self.store_param_remainders:
            self._local_step_with_param_remainders(bucket_ids)
            return

        # Find param fragments for each bucket
        buffers = collections.defaultdict(list) # p_in, m, v, g, p_out
        for bucket_id in bucket_ids:

            # Optimizer state buffers for local shard
            fragments = self.state['buckets'][bucket_id].fragments
            exp_avg = self.state['buckets'][bucket_id].exp_avg_shard
            exp_avg_sq = self.state['buckets'][bucket_id].exp_avg_sq_shard
            grads = self._grads_buckets[bucket_id].grads_shard
            params_out = self._params_buckets[bucket_id].params_shard

            # Find param fragments in local shard
            for fragment in fragments:
                if fragment.in_local_shard:
                    param_group_id = fragment.param_group_id
                    shard_start, shard_end = fragment.shard_range
                    if self.store_params:
                        params_shard = self.state['buckets'][bucket_id].params_shard
                        param_fragment = params_shard[shard_start:shard_end]
                    else:
                        param_id = fragment.param_id
                        param = self.param_groups[param_group_id]['params'][param_id]
                        param_start, param_end = fragment.shard_param_range
                        param_fragment = param.detach().view(-1)[param_start:param_end]
                        param_fragment = param_fragment.to(dtype=self.dtype, device=self.device)
                    if shard_end > shard_start:
                        buffers[param_group_id].append([
                            param_fragment,
                            exp_avg[shard_start:shard_end],
                            exp_avg_sq[shard_start:shard_end],
                            grads[shard_start:shard_end],
                            params_out[shard_start:shard_end],
                        ])

        # Apply optimizer step to each param group
        for group_id, group_buffers in buffers.items():
            group = self.param_groups[group_id]
            beta1, beta2 = group['betas']
            multi_tensor_applier(
                distributed_adam_cuda.multi_tensor_fused_adam,
                self._dummy_overflow_buf,
                list(zip(*group_buffers)),
                self._grad_scale,
                group['lr'],
                beta1,
                beta2,
                group['eps'],
                self.state['step'],
                1 if self.adam_w_mode else 0,
                1 if group['bias_correction'] else 0,
                group['weight_decay'],
            )

    def _local_step_with_param_remainders(self, bucket_ids):
        """Apply optimizer step to local shard of parameter bucket

        This is an experimental implementation that expects
        store_params=False and store_param_remainders=True. The
        optimizer dtype must be FP32 and the params must all be BF16
        and GPU.

        Arguments:
            bucket_ids (iterable): bucket indices

        """

        # Find param fragments for each bucket
        buffers = collections.defaultdict(list) # p_in, p_rem, m, v, g, p_out
        for bucket_id in bucket_ids:

            # State buffers for local shard
            fragments = self.state['buckets'][bucket_id].fragments
            param_remainders_shard = self.state['buckets'][bucket_id].param_remainders_shard
            exp_avg = self.state['buckets'][bucket_id].exp_avg_shard
            exp_avg_sq = self.state['buckets'][bucket_id].exp_avg_sq_shard
            grads = self._grads_buckets[bucket_id].grads_shard
            params_out = self._params_buckets[bucket_id].params_shard

            # Find param fragments in local shard
            for fragment in fragments:
                if fragment.in_local_shard:
                    param_group_id = fragment.param_group_id
                    param_id = fragment.param_id
                    param_start, param_end = fragment.shard_param_range
                    shard_start, shard_end = fragment.shard_range
                    param = self.param_groups[param_group_id]['params'][param_id]
                    param_fragment = param.detach().view(-1)[param_start:param_end]
                    param_fragment = param_fragment.to(dtype=torch.bfloat16, device=self.device)
                    if shard_end > shard_start:
                        buffers[param_group_id].append([
                            param_fragment,
                            param_remainders_shard[shard_start:shard_end],
                            exp_avg[shard_start:shard_end],
                            exp_avg_sq[shard_start:shard_end],
                            grads[shard_start:shard_end],
                            params_out[shard_start:shard_end],
                        ])

        # Apply optimizer step to each param group
        for group_id, group_buffers in buffers.items():
            group = self.param_groups[group_id]
            beta1, beta2 = group['betas']
            multi_tensor_applier(
                distributed_adam_cuda.multi_tensor_fused_adam_with_param_remainders,
                self._dummy_overflow_buf,
                list(zip(*group_buffers)),
                self._grad_scale,
                group['lr'],
                beta1,
                beta2,
                group['eps'],
                self.state['step'],
                1 if self.adam_w_mode else 0,
                1 if group['bias_correction'] else 0,
                group['weight_decay'],
            )

    def state_dict(self, gather_on_root=True):
        """Get dictionary containing optimizer state

        Default behavior is to perform communication so that the
        entire optimizer state is returned on the root rank in the
        process group. In this case, all ranks in the process group
        must enter this function and no value is returned on non-root
        ranks.

        Arguments:
            gather_on_root (bool, optional): Gather state from all
                ranks on the root rank (default: True)

        """
        state_dict = super().state_dict()
        if not gather_on_root:
            return state_dict

        # Finish any asynchronous communication
        self.grad_sync()
        self.param_sync()

        # Export local state to byte string
        state_bytes = io.BytesIO()
        torch.save(state_dict, state_bytes)
        state_bytes.seek(0)
        state_bytes_view = state_bytes.getbuffer()

        # Get data sizes on all ranks
        local_state_size = len(state_bytes_view)
        state_sizes = [None] * self.distributed_size
        torch.distributed.all_gather_object(
            state_sizes,
            local_state_size,
            group=self.process_group,
        )
        max_state_size = max(state_sizes)

        # Construct workspace buffers
        chunk_size = self.default_shard_size * torch.finfo(self.grad_sync_dtype).bits // 8
        if self.distributed_rank == 0:
            gathered_state_bytes = [
                torch.empty([size], dtype=torch.uint8, device='cpu')
                for size in state_sizes
            ]
            gathered_state_bytes[0].copy_(
                torch.frombuffer(state_bytes_view, dtype=torch.uint8)
            )
            gathered_chunks_buffers = [
                torch.empty(
                    [chunk_size * self.distributed_size],
                    dtype=torch.uint8,
                    device=self.device,
                )
                for _ in range(self.pipeline_size)
            ]
        else:
            chunk_buffers = [
                torch.empty(
                    [chunk_size],
                    dtype=torch.uint8,
                    device=self.device,
                )
                for _ in range(self.pipeline_size)
            ]

        # Split data into chunks and gather on root rank
        # Note: Assuming we are using the NCCL backend, communication
        # must happen on the GPU. We split the data into fixed-size
        # chunks to limit GPU memory usage.
        # TODO: Avoid chunking with direct communication between CPUs
        main_stream = torch.cuda.current_stream()
        for stream in self._pipeline_streams:
            stream.wait_stream(main_stream)
        for stream_id, offset in enumerate(range(0, max_state_size, chunk_size)):
            stream_id %= self.pipeline_size
            stream = self._pipeline_streams[stream_id]
            with torch.cuda.stream(stream):

                # Buffers for chunk
                if self.distributed_rank == 0:
                    gathered_chunks = [
                        gathered_chunks_buffers[stream_id][i*chunk_size:(i+1)*chunk_size]
                        for i in range(self.distributed_size)
                    ]
                else:
                    chunk = chunk_buffers[stream_id]

                # Copy to GPU
                if self.distributed_rank != 0 and offset < local_state_size:
                    local_chunk_size = min(chunk_size, local_state_size-offset)
                    chunk[:local_chunk_size].copy_(
                        torch.frombuffer(
                            state_bytes_view,
                            dtype=torch.uint8,
                            count=local_chunk_size,
                            offset=offset,
                        ),
                        non_blocking=True,
                    )

                # Gather on root
                # Note: Call in main stream to avoid memory pool
                # overheads from internal memory allocations in
                # gather.
                main_stream.wait_stream(stream)
                with torch.cuda.stream(main_stream):
                    if self.distributed_rank == 0:
                        if self._gather_no_copy:
                            no_copy_kwarg = { 'no_copy': True }
                        else:
                            no_copy_kwarg = {}
                        torch.distributed.gather(
                            gathered_chunks[0],
                            gathered_chunks,
                            dst=self.process_group_root,
                            group=self.process_group,
                            **no_copy_kwarg,
                        )
                    else:
                        torch.distributed.gather(
                            chunk,
                            dst=self.process_group_root,
                            group=self.process_group,
                        )
                stream.wait_stream(main_stream)

                # Copy back to CPU
                if self.distributed_rank == 0:
                    for rank in range(1, self.distributed_size):
                        rank_chunk_start = offset
                        rank_chunk_end = min(offset + chunk_size, state_sizes[rank])
                        rank_chunk_size = rank_chunk_end - rank_chunk_start
                        if rank_chunk_size > 0:
                            src = gathered_chunks[rank][:rank_chunk_size]
                            dst = gathered_state_bytes[rank][rank_chunk_start:rank_chunk_end]
                            dst.copy_(src, non_blocking=True)

        # Synchronize GPU
        for stream in self._pipeline_streams:
            main_stream.wait_stream(stream)
        main_stream.synchronize()

        # Return gathered state data on root rank
        if self.distributed_rank == 0:
            return {'gathered_states': gathered_state_bytes}
        else:
            return None

    def load_state_dict(self, state_dict):
        """Load optimizer state"""

        # State dict contains state for all ranks
        if 'gathered_states' in state_dict:

            # Deallocate distributed optimizer state to reduce GPU
            # memory usage
            if 'buckets' in self.state:
                del self.state['buckets']

            # Get state for current rank and parse byte string
            state_bytes = state_dict['gathered_states'][self.distributed_rank]
            state_bytes = io.BytesIO(state_bytes.numpy())
            state_dict = torch.load(state_bytes)

        return super().load_state_dict(state_dict)
