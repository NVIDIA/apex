import collections
import contextlib
import enum
import importlib
import inspect
import math
import threading

import torch
import amp_C
from apex.multi_tensor_apply import multi_tensor_applier
from torch.distributed.distributed_c10d import _get_default_group, _get_global_rank

class DistributedFusedAdam(torch.optim.Optimizer):
    """AdamW optimizer with ZeRO algorithm.

    Currently GPU-only. Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

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
        betas (Tuple[float, float], optional): coefficients used for
            computing running averages of gradient and its square.
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
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
            parameter synchronization (default: same as
            grad_sync_dtype)
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
        bucket_cap_mb (float, optional): bucket size in megabytes
            (default: 100)
        pipeline_size (int, optional): number of buckets to
            synchronize simultaneously (default: 2)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
    .. _ZeRO\: Memory Optimizations Toward Training Trillion Parameter Models:
        https://arxiv.org/abs/1910.02054

    """

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

    _step_supports_amp_scaling = True

    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
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
                 bucket_cap_mb=100,
                 pipeline_size=2,
    ):
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(DistributedFusedAdam, self).__init__(params, defaults)

        # Adam options
        if amsgrad:
            raise RuntimeError('DistributedFusedAdam does not support the AMSGrad variant.')

        # Datatype options
        if grad_sync_dtype is None:
            grad_sync_dtype = dtype
        if param_sync_dtype is None:
            param_sync_dtype = grad_sync_dtype
        supported_dtypes = [
            (torch.float32, torch.float16, torch.float16),
            (torch.float32, torch.float32, torch.float32),
        ]
        if (dtype, grad_sync_dtype, param_sync_dtype) not in supported_dtypes:
            raise RuntimeError(
                'Invalid dtypes for DistributedFusedAdam '
                f'(dtype={dtype}, '
                f'grad_sync_dtype={grad_sync_dtype}, '
                f'param_sync_dtype={param_sync_dtype}))')
        if device != 'cuda':
            raise RuntimeError('DistributedFusedAdam only supports GPU')
        self.dtype = dtype
        self.grad_sync_dtype = grad_sync_dtype
        self.param_sync_dtype = param_sync_dtype
        self.device = device

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

        # Use average reduction for grad sync
        self.average_grad_sync = average_grad_sync
        # Copy param grads to bucket as soon as available
        self.greedy_grad_copy = True
        # Synchronize grad buckets as soon as all grads are available
        self.overlap_grad_sync = overlap_grad_sync
        # Number of buckets to synchronize at a time
        self.pipeline_size = pipeline_size

        # Determine bucket sizes
        dtype_size = torch.finfo(self.grad_sync_dtype).bits // 8
        self.alignment = 128 // dtype_size
        bucket_size = 1024*1024*bucket_cap_mb / dtype_size
        shard_size = bucket_size / self.distributed_size
        shard_size = (int(shard_size) // self.alignment) * self.alignment
        shard_size = max(shard_size, self.alignment)
        bucket_size = shard_size * self.distributed_size
        self.bucket_size = bucket_size
        self.shard_size = shard_size

        # Load CUDA kernels
        global fused_adam_cuda, distributed_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")
        distributed_adam_cuda = importlib.import_module("distributed_adam_cuda")

        # Optimizer state
        self.state['buckets'] = []
        self.state['step'] = 0

        # Objects for gradient synchronization
        self._grads_generated = set()
        self._pipeline_streams = [torch.cuda.Stream() for _ in range(self.pipeline_size)]

        # Divide gradients by factor before optimizer step. Used for
        # grad clipping and gradient scaler.
        self._inv_grad_scale = torch.full([1], 1.0, dtype=self.dtype, device=self.device)
        # Norm of parameter gradients. Used for gradient clipping and
        # gradient scaler.
        self._grad_norm = None

        # Check if collectives have no_copy option
        self._reduce_scatter_no_copy = (
            'no_copy' in inspect.getfullargspec(torch.distributed.reduce_scatter).args
        )
        self._all_gather_no_copy = (
            'no_copy' in inspect.getfullargspec(torch.distributed.all_gather).args
        )

        # Attach hooks for gradient synchronization
        self._register_post_backward_hooks()

    def _register_post_backward_hooks(self):
        """Attach hooks for gradient synchronization

        Optimizer state for parameters are initialized lazily as they
        are encountered in the backward pass.

        """
        self._num_grads = 0
        self._lock = threading.Lock()
        self._grad_accs = []
        try:
            root_rank = _get_global_rank(self.process_group, 0)
        except:
            root_rank = 0
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group['params']):
                torch.distributed.broadcast(
                    param,
                    src=root_rank,
                    group=self.process_group,
                )
                if param.requires_grad:
                    def wrapper(p, p_group_id, p_id):
                        p_tmp = p.expand_as(p)
                        grad_acc = p_tmp.grad_fn.next_functions[0][0]
                        def reduction_hook(*unused):
                            with self._lock:
                                if 'fragments' not in self.state[p]:
                                    self._init_param_state(p, p_group_id, p_id)
                                if self.greedy_grad_copy:
                                    self._grad_copy(p)
                                    if self.overlap_grad_sync:
                                        self._try_start_bucket_grad_sync(
                                            [p],
                                            ignore_last_bucket=True,
                                        )
                        grad_acc.register_hook(reduction_hook)
                        self._grad_accs.append(grad_acc)
                    wrapper(param, param_group_id, param_id)
                    self._num_grads += 1

    def _init_param_state(
            self,
            param,
            param_group_id,
            param_id,
    ):
        """Initialize optimizer state for a parameter"""

        # Make sure there is at least one bucket
        if not self.state['buckets']:
            self._add_bucket()

        # Split parameter values into fragments
        # Note: Each fragment resides within a bucket
        param_start = 0
        param_size = param.numel()
        self.state[param]['fragments'] = []
        while param_start < param_size:

            # Get current bucket
            if not self.state['buckets']:
                self._add_bucket()
            bucket_id = len(self.state['buckets']) - 1
            bucket = self.state['buckets'][bucket_id]
            fragment_id = len(bucket['fragments'])

            # Determine fragment position within bucket
            if fragment_id == 0:
                bucket_start = 0
            else:
                bucket_start = bucket['fragments'][-1]['bucket_range'][1]
                bucket_start = (
                    (bucket_start + self.alignment - 1)
                    // self.alignment
                    * self.alignment
                ) # Pad until fragment is aligned
            fragment_size = min(param_size-param_start, self.bucket_size-bucket_start)
            param_end = param_start + fragment_size
            bucket_end = bucket_start + fragment_size

            # Create new bucket if current one is full
            if fragment_size <= 0:
                self._add_bucket()
                continue

            # Fragment position within local shard
            shard_id = self.distributed_rank
            shard_start = bucket_start - self.shard_size*shard_id
            shard_end = bucket_end - self.shard_size*shard_id
            shard_start = min(max(shard_start, 0), self.shard_size)
            shard_end = min(max(shard_end, 0), self.shard_size)
            in_local_shard = shard_start < shard_end
            if in_local_shard:
                shard_bucket_start = shard_start + self.shard_size*shard_id
                shard_bucket_end = shard_bucket_start + shard_end - shard_start
                shard_param_start = shard_bucket_start - bucket_start + param_start
                shard_param_end = shard_param_start + shard_end - shard_start
            else:
                shard_bucket_start, shard_bucket_end = None, None
                shard_param_start, shard_param_end = None, None

            # Record fragment info
            fragment = {
                # Parameter group index
                'param_group_id': param_group_id,
                # Parameter index within parameter group
                'param_id': param_id,
                # Bucket index
                'bucket_id': bucket_id,
                # Range within flattened parameter buffer
                'param_range': (param_start,param_end),
                # Range within bucket
                'bucket_range': (bucket_start,bucket_end),
                # Whether fragment is in local shard of bucket
                'in_local_shard': in_local_shard,
                # Range within local shard
                'shard_range': (shard_start,shard_end),
                # Range of local fragment shard within bucket
                'shard_bucket_range': (shard_bucket_start,shard_bucket_end),
                # Range of local fragment shard within parameter
                'shard_param_range': (shard_param_start,shard_param_end),
            }

            # Record fragment info
            self.state[param]['fragments'].append(fragment)
            bucket['fragments'].append(fragment)
            param_start = param_end

        # Initialize master param buffer
        for fragment in self.state[param]['fragments']:
            if fragment['in_local_shard']:
                bucket_id = fragment['bucket_id']
                bucket = self.state['buckets'][bucket_id]
                param_start, param_end = fragment['shard_param_range']
                shard_start, shard_end = fragment['shard_range']
                model_param_fragment = param.view(-1)[param_start:param_end]
                master_param_fragment = bucket['params_shard'][shard_start:shard_end]
                master_param_fragment.copy_(model_param_fragment)

    def _add_bucket(self):
        """Construct a bucket for optimizer state"""
        self.state['buckets'].append({

            # Parameter fragments associated with bucket
            'fragments': [],

            # Gradient buffers
            'grads_shard': None,
            'grads_bucket': None,
            'curr_grads_shard': None,   # For current micro-batch

            # Optimizer state
            'params_shard': torch.zeros([self.shard_size], dtype=self.dtype, device=self.device),
            'exp_avg_shard': torch.zeros([self.shard_size], dtype=self.dtype, device=self.device),
            'exp_avg_sq_shard': torch.zeros([self.shard_size], dtype=self.dtype, device=self.device),

            # Status of parameter gradients
            'gradient_status': self.GradientStatus.READY,

            # Distributed request object for gradient synchronization
            'grad_sync_request': None,

        })

    def zero_grad(self, set_to_none=True):
        """Clear parameter gradients"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None or set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
        for bucket in self.state['buckets']:
            bucket['grads_shard'] = None
            bucket['grads_bucket'] = None
            bucket['curr_grads_shard'] = None
            bucket['gradient_status'] = self.GradientStatus.READY
        self._grads_generated = set()
        self._inv_grad_scale = torch.full([1], 1.0, dtype=self.dtype, device=self.device)
        self._grad_norm = None

    def _grad_copy(self, param):
        """Copy parameter gradients to buckets"""

        # Copy param grad to buckets
        for fragment in self.state[param]['fragments']:

            # Get fragment position
            bucket_id = fragment['bucket_id']
            bucket = self.state['buckets'][bucket_id]
            grad_start, grad_end = fragment['param_range']
            bucket_start, bucket_end = fragment['bucket_range']

            # Set reduction status
            if bucket['gradient_status'] == self.GradientStatus.SYNCING:
                self._finish_bucket_grad_sync()
            bucket['gradient_status'] = self.GradientStatus.PARTIALLY_FILLED

            # Allocate gradient buffer if needed
            if bucket['grads_bucket'] is None:
                bucket['grads_bucket'] = torch.zeros(
                    [self.bucket_size],
                    dtype=self.grad_sync_dtype,
                    device=self.device,
                )

            # Copy param grad to bucket
            if param.grad is not None:
                scale = 1/self.process_group_size if self.average_grad_sync else 1.0
                grad_in = param.grad.detach().view(-1)[grad_start:grad_end]
                grad_out = bucket['grads_bucket'][bucket_start:bucket_end]
                grad_out.add_(grad_in, alpha=scale)

        # Free param grad buffer
        param.grad = None

    def _force_bucket_grad_sync(self):
        """Ensure that all gradient buckets are synchronized"""

        # Synchronize all unsynchronized buckets
        self._finish_bucket_grad_sync()
        buckets = [
            bucket for bucket in self.state['buckets']
            if bucket['gradient_status'] != self.GradientStatus.READY
        ]
        if buckets:
            self._start_bucket_grad_sync(buckets)
            self._finish_bucket_grad_sync()

        # Fill any unfilled buckets with zeros
        for bucket in self.state['buckets']:
            if bucket['grads_shard'] is None:
                bucket['grads_shard'] = torch.zeros(
                    [self.shard_size],
                    dtype=self.grad_sync_dtype,
                    device=self.device,
                )

        # Reset set of generated gradients
        self._grads_generated = set()

    def _try_start_bucket_grad_sync(
            self,
            params=[],
            ignore_last_bucket=True,
    ):
        """Launches gradient synchronization if enough buckets are ready

        Gradient synchronization is asynchronous. Launches gradient
        synchronization if all gradients have been generated or if
        there are enough buckets ready to fill pipeline.

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
            self._grads_generated.add(param)
            for fragment in self.state[param]['fragments']:
                bucket_id = fragment['bucket_id']
                bucket = self.state['buckets'][bucket_id]
                is_filled = True
                for other_fragment in reversed(bucket['fragments']):
                    param_group_id = other_fragment['param_group_id']
                    param_id = other_fragment['param_id']
                    other_param = self.param_groups[param_group_id]['params'][param_id]
                    if other_param not in self._grads_generated:
                        is_filled = False
                        break
                if is_filled:
                    bucket['gradient_status'] = self.GradientStatus.FULLY_FILLED

        # Launch reductions if enough buckets are ready
        if len(self._grads_generated) == self._num_grads:
            self._force_bucket_grad_sync()
        else:
            all_buckets = self.state['buckets']
            if ignore_last_bucket:
                all_buckets = all_buckets[:-1]
            filled_buckets = [
                bucket
                for bucket in all_buckets
                if bucket['gradient_status'] == self.GradientStatus.FULLY_FILLED
            ]
            pipeline_size = (len(filled_buckets) // self.pipeline_size) * self.pipeline_size
            if pipeline_size > 0:
                self._start_bucket_grad_sync(filled_buckets[:pipeline_size])

    def _start_bucket_grad_sync(self, buckets):
        """Synchronize gradients in buckets

        Gradient synchronization is asynchronous. Involves
        reduce-scatter over distributed process group and allreduce
        over redundant process group.

        """
        self._finish_bucket_grad_sync()

        # Reduce gradients
        for stream in self._pipeline_streams:
            stream.wait_stream(torch.cuda.current_stream())
        for i, bucket in enumerate(buckets):
            bucket['gradient_status'] = self.GradientStatus.SYNCING
            stream = self._pipeline_streams[i % self.pipeline_size]
            with torch.cuda.stream(stream):

                # Reduce-scatter over distributed process group
                if self.distributed_size == 1:
                    bucket['curr_grads_shard'] = bucket['grads_bucket']
                    bucket['grad_sync_request'] = None
                else:
                    bucket['curr_grads_shard'] = torch.zeros(
                        [self.shard_size],
                        dtype=self.grad_sync_dtype,
                        device=self.device,
                    )
                    grads_bucket_shards = [
                        bucket['grads_bucket'][i*self.shard_size:(i+1)*self.shard_size]
                        for i in range(self.distributed_size)
                    ]
                    if self._reduce_scatter_no_copy:
                        no_copy_kwarg = { 'no_copy': True }
                    else:
                        no_copy_kwarg = {}
                    bucket['grad_sync_request'] = (
                        torch.distributed.reduce_scatter(
                            bucket['curr_grads_shard'],
                            grads_bucket_shards,
                            group=self.distributed_process_group,
                            async_op=True,
                            **no_copy_kwarg,
                        )
                    )

                # All-reduce over redundant process group
                # Note: Assuming reduce-scatters are finished in the
                # order they are submitted, all-reduces should be
                # submitted in a consistent order. There could be race
                # conditions if wait doesn't finish in order.
                if self.redundant_size > 1:
                    if bucket['grad_sync_request'] is not None:
                        bucket['grad_sync_request'].wait()
                    bucket['grad_sync_request'] = (
                        torch.distributed.all_reduce(
                            bucket['curr_grads_shard'],
                            group=self.redundant_process_group,
                            async_op=True,
                        )
                    )

    def _finish_bucket_grad_sync(self):
        """Wait for any gradient synchronizations that are in progress"""
        for bucket in self.state['buckets']:
            if bucket['gradient_status'] == self.GradientStatus.SYNCING:

                # Finish asynchronous communication
                if bucket['grad_sync_request'] is not None:
                    bucket['grad_sync_request'].wait()
                bucket['grad_sync_request'] = None

                # Accumulate gradient in local shard
                if bucket['grads_shard'] is None:
                    bucket['grads_shard'] = bucket['curr_grads_shard']
                else:
                    bucket['grads_shard'].add_(bucket['curr_grads_shard'])

                # Deallocate buffers for gradient synchronization
                bucket['grads_bucket'] = None
                bucket['curr_grads_shard'] = None

                # Reset status
                bucket['gradient_status'] = self.GradientStatus.READY

                # Cached gradient norm has been invalidated
                self._grad_norm = None

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
            for fragment in bucket['fragments']:
                param_group_id = fragment['param_group_id']
                param_id = fragment['param_id']
                param = self.param_groups[param_group_id]['params'][param_id]
                if param.grad is not None:
                    self._grad_copy(param)
                    self._try_start_bucket_grad_sync(
                        [param],
                        ignore_last_bucket=False,
                    )
        self._force_bucket_grad_sync()

    def _local_grad_norm(self, parameters=[], norm_type=2.0):
        """Local contribution to parameter gradient norm

        Returns square of 2-norm. Other norms are not yet supported.

        If no parameters are provided, the norm is computed for all
        parameters in optimizer. Provided parameters are assumed to be
        in optimizer.

        """
        norm_type = float(norm_type)
        assert norm_type == 2.0

        # Make sure that gradients have been reduced
        self.grad_sync()

        if not parameters or len(parameters) == self._num_grads:
            # Compute norm of all local gradients
            dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
            grad_norm_sq = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [[bucket['grads_shard'] for bucket in self.state['buckets']]],
                False,
            )[0] ** 2
        else:
            # Compute norm of selected local gradients
            grads = []
            for param in parameters:
                for fragment in self.state[param]['fragments']:
                    if fragment['in_local_shard']:
                        bucket_id = fragment['bucket_id']
                        bucket = self.state['buckets'][bucket_id]
                        shard_start, shard_end = fragment['shard_range']
                        grads.append(bucket['grads_shard'][shard_start:shard_end])
            if grads:
                dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
                grad_norm_sq = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads],
                    False,
                )[0] ** 2
            else:
                grad_norm_sq = torch.zeros([1], dtype=torch.float32, device=self.device)

        return grad_norm_sq.detach().view([])

    def grad_norm(self, parameters=[], norm_type=2.0, force=False):
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
        return self._grad_norm.detach()

    def clip_grad_norm(self, max_norm, parameters=[], norm_type=2.0):
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
        inv_clip_coef = (total_norm + 1e-6) / max_norm
        self._inv_grad_scale = torch.clamp(inv_clip_coef, min=1.0).view(1)
        return total_norm

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

        # Make sure that gradients have been reduced
        self.grad_sync()

        # Apply gradient scaler if provided
        # Note: We compute gradient norm to check for non-finite
        # values. This is more conservative and compute intensive than
        # directly checking, but it avoids extra communication if we
        # have already computed gradient norm e.g. for gradient
        # clipping.
        if grad_scaler is not None:
            grad_norm = self.grad_norm()
            found_inf = torch.logical_not(torch.isfinite(grad_norm))
            scaler_state = grad_scaler._per_optimizer_states[id(self)]
            scaler_state['found_inf_per_device'] = {found_inf.device: found_inf.float()}
            if found_inf.item():
                return
            else:
                assert grad_scaler._scale is not None
                self._inv_grad_scale *= grad_scaler._scale
        inv_grad_scale = self._inv_grad_scale.item()

        # Apply optimizer step to each bucket and synchronize params
        self.state['step'] += 1
        current_stream = torch.cuda.current_stream()
        for stream in self._pipeline_streams:
            stream.wait_stream(current_stream)
        for i, bucket in enumerate(self.state['buckets']):
            stream = self._pipeline_streams[i % self.pipeline_size]
            with torch.cuda.stream(stream):

                # Buffer for param sync
                params_shard_copy = torch.zeros(
                    [self.shard_size],
                    dtype=self.param_sync_dtype,
                    device=self.device,
                )

                # Find param fragments in local shard
                buffers = collections.defaultdict(list) # p, m, v, g, p_copy
                for fragment in bucket['fragments']:
                    if fragment['in_local_shard']:
                        param_group_id = fragment['param_group_id']
                        shard_start, shard_end = fragment['shard_range']
                        buffers[param_group_id].append([
                            bucket['params_shard'][shard_start:shard_end],
                            bucket['exp_avg_shard'][shard_start:shard_end],
                            bucket['exp_avg_sq_shard'][shard_start:shard_end],
                            bucket['grads_shard'][shard_start:shard_end],
                            params_shard_copy[shard_start:shard_end],
                        ])

                # Fuse param fragments if possible
                if len(buffers) == 1:
                    group_id = list(buffers.keys())[0]
                    buffers[group_id] = [(
                        bucket['params_shard'],
                        bucket['exp_avg_shard'],
                        bucket['exp_avg_sq_shard'],
                        bucket['grads_shard'],
                        params_shard_copy,
                    )]

                # Apply optimizer step to each param group
                for group_id, group_buffers in buffers.items():

                    # Get param group configs
                    group = self.param_groups[group_id]
                    beta1, beta2 = group['betas']
                    bias_correction = 1 if group['bias_correction'] else 0
                    eps = group['eps']
                    weight_decay = group['weight_decay']

                    # Copy param group configs to GPU
                    num_fragments = len(group_buffers)
                    beta1 = torch.full([num_fragments], beta1, dtype=self.dtype, device='cuda')
                    beta2 = torch.full([num_fragments], beta2, dtype=self.dtype, device='cuda')
                    bias_correction = torch.full([num_fragments], bias_correction, dtype=torch.int32, device='cuda')
                    eps = torch.full([num_fragments], eps, dtype=self.dtype, device='cuda')
                    weight_decay = torch.full([num_fragments], weight_decay, dtype=self.dtype, device='cuda')

                    # Apply Adam step
                    dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
                    multi_tensor_applier(
                        distributed_adam_cuda.multi_tensor_fused_adam,
                        dummy_overflow_buf,
                        list(zip(*group_buffers)),
                        beta1,
                        beta2,
                        bias_correction,
                        eps,
                        weight_decay,
                        group['lr'],
                        inv_grad_scale,
                        self.state['step'],
                        1, # Set to 0 to apply eps inside sqrt
                    )
                    del group_buffers

                # Deallocate buffers
                del buffers

                # Allgather updated parameters
                if self.distributed_size == 1:
                    params_bucket = params_shard_copy
                else:
                    params_bucket = torch.zeros(
                        [self.bucket_size],
                        dtype=self.param_sync_dtype,
                        device=self.device,
                    )
                    params_bucket_shards = [
                        params_bucket[i*self.shard_size:(i+1)*self.shard_size]
                        for i in range(self.distributed_size)
                    ]
                    params_bucket_shards[self.distributed_rank].copy_(params_shard_copy)
                    if self._all_gather_no_copy:
                        no_copy_kwarg = { 'no_copy': True }
                    else:
                        no_copy_kwarg = {}
                    torch.distributed.all_gather(
                        params_bucket_shards,
                        params_bucket_shards[self.distributed_rank],
                        group=self.distributed_process_group,
                        **no_copy_kwarg,
                    )
                del params_shard_copy

                # Copy values to param buffers
                buffers = collections.defaultdict(list) # param_in, param_out
                for fragment in bucket['fragments']:
                    param_group_id = fragment['param_group_id']
                    param_id = fragment['param_id']
                    param = self.param_groups[param_group_id]['params'][param_id]
                    bucket_start, bucket_end = fragment['bucket_range']
                    param_start, param_end = fragment['param_range']
                    buffers[(param.is_cuda, param.dtype)].append((
                        params_bucket[bucket_start:bucket_end],
                        param.detach().view(-1)[param_start:param_end],
                    ))
                for (is_cuda, dtype), dtype_buffers in buffers.items():
                    fused_kernel_dtypes = (torch.float32, torch.float16, torch.uint8)
                    if (is_cuda
                        and dtype in fused_kernel_dtypes
                        and self.param_sync_dtype in fused_kernel_dtypes):
                        dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
                        multi_tensor_applier(
                            fused_adam_cuda.maybe_cast_mt,
                            dummy_overflow_buf,
                            list(zip(*dtype_buffers)),
                        )
                    else:
                        for param_in, param_out in dtype_buffers:
                            param_out.copy_(param_in)
                    del dtype_buffers
                del params_bucket, buffers

        # Synchronize pipeline streams
        for stream in self._pipeline_streams:
            current_stream.wait_stream(stream)

        return loss
