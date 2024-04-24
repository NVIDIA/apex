import collections
import contextlib
from dataclasses import dataclass
import enum
import inspect
import io
import itertools
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import warnings

import torch
from torch.distributed.distributed_c10d import _get_default_group

try:
    import apex.contrib.nccl_allocator as nccl_allocator
except ImportError:
    nccl_allocator = None

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

# Import context manager to coalesce NCCL calls
# Note: Replace these backward compatibility shims once PyTorch
# exposes a stable public API for coalescing communication.
from torch.distributed.distributed_c10d import _coalescing_manager

if "device" not in inspect.signature(_coalescing_manager).parameters:
    # PyTorch <=1.13.1 does not have device arg
    _coalescing_manager_no_device_arg = _coalescing_manager

    @contextlib.contextmanager
    def _coalescing_manager(group, device, reqs):
        with _coalescing_manager_no_device_arg(group, reqs):
            yield


if "reqs" in inspect.signature(_coalescing_manager).parameters:
    # PyTorch <=2.0.1 handles synchronization externally to coalescing
    # manager
    _coalescing_manager_with_reqs_arg = _coalescing_manager

    class _CoalescingManager:
        def __init__(self):
            self.works: List[torch.distributed.Work] = []

        def append(self, work: torch.distributed.Work) -> None:
            if work:
                self.works.append(work)

        def wait(self) -> None:
            for work in self.works:
                work.wait()

    @contextlib.contextmanager
    def _coalescing_manager(
        group: Optional[torch.distributed.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        async_ops: bool = False,
    ) -> contextlib.AbstractContextManager:
        assert device is not None
        cm = _CoalescingManager()
        with _coalescing_manager_with_reqs_arg(
            group,
            device,
            cm.works,
        ):
            yield cm
        if not async_ops:
            cm.wait()

    def _coalescing_manager_append_work(
        cm: _CoalescingManager,
        work: torch.distributed.Work,
    ) -> None:
        """Add asynchronous request to coalescing manager"""
        cm.append(work)

else:
    # PyTorch >2.0.1 handles synchronization within coalescing
    # manager
    def _coalescing_manager_append_work(
        cm: torch.distributed._CoalescingManager,
        work: torch.distributed.Work,
    ) -> None:
        """Dummy function for backward compatibility

        Coalescing manager already keeps track of asynchronous
        communication.

        """
        pass


# Import optional CUDA kernels
_FOUND_DEPRECATED_FUSED_ADAM: bool = False
try:
    import fused_adam_cuda

    _FOUND_DEPRECATED_FUSED_ADAM = True
except ImportError:
    warnings.warn(
        "Could not find recommended CUDA kernels when importing "
        "`DistributedFusedAdam`. "
        "For best performance, Apex should be installed with "
        "`--deprecated_fused_adam`."
    )


def _round_to_multiple(
    number: int,
    multiple: int,
    round_up: bool = True,
) -> int:
    """Assumes arguments are positive integers"""
    return (number + multiple - 1 if round_up else number) // multiple * multiple


def _devices_match(device1: torch.device, device2: torch.device) -> bool:
    """Whether two PyTorch devices are equivalent"""
    device1 = torch.device(device1)
    device2 = torch.device(device2)
    if device1.type != device2.type:
        return False
    if device1.type == "cuda":
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
    buffers_in: List[torch.Tensor],
    buffers_out: List[torch.Tensor],
    dummy_overflow_buf: Optional[torch.Tensor] = None,
) -> None:
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
        is_cuda = (
            _devices_match(buf_in.device, "cuda")
            and _devices_match(buf_out.device, "cuda")
        )
        is_contiguous = buf_in.is_contiguous() and buf_out.is_contiguous()
        key = (
            buf_in.dtype,
            buf_out.dtype,
            is_cuda,
            is_contiguous,
        )
        buffer_groups[key].append((buf_in, buf_out))

    # Copy each group of buffers
    for key, buffers in buffer_groups.items():
        # Check if buffers support fused kernel
        dtype_in, dtype_out, is_cuda, is_contiguous = key
        supported_dtypes = (torch.float32, torch.float16)
        use_fused_kernel = (
            dtype_in in supported_dtypes and dtype_out in supported_dtypes
        ) or (dtype_in == torch.uint8 and dtype_out == torch.uint8)
        use_fused_kernel = use_fused_kernel and is_cuda and is_contiguous

        # Copy buffers
        if use_fused_kernel and _FOUND_DEPRECATED_FUSED_ADAM:
            if dummy_overflow_buf is None:
                dummy_overflow_buf = torch.zeros([1], dtype=torch.int32, device="cuda")
            multi_tensor_applier(
                fused_adam_cuda.maybe_cast_mt,
                dummy_overflow_buf,
                list(zip(*buffers)),
            )
        else:
            for buf_in, buf_out in buffers:
                buf_out.copy_(buf_in)


@contextlib.contextmanager
def _disable_pre_forward_hook(
    param: torch.nn.Parameter,
) -> contextlib.AbstractContextManager:
    """Prevent parameter from calling pre-forward hook"""
    hook_is_enabled = getattr(
        param,
        "_pre_forward_hook_is_enabled",
        False,
    )
    if hook_is_enabled:
        param._pre_forward_hook_is_enabled = False
    try:
        yield
    finally:
        if hook_is_enabled:
            param._pre_forward_hook_is_enabled = True


@torch.no_grad()
def _bf16_rem_to_fp32(
    bf16: torch.Tensor,
    rem: torch.Tensor,
    fp32: torch.Tensor,
) -> None:
    """Pack BF16 tensor and 16-bit remainders into FP32 tensor"""

    # Check inputs
    assert bf16.size() == rem.size() == fp32.size(), (
        "Tensor dimensions do not match: "
        f"bf16={list(bf16.size())}, "
        f"rem={list(rem.size())}, "
        f"fp32={list(fp32.size())}, "
    )
    assert bf16.dtype is torch.bfloat16, f"bf16 buffer has invalid dtype ({bf16.dtype})"
    assert rem.dtype is torch.int16, f"rem buffer has invalid dtype ({rem.dtype})"
    assert fp32.dtype is torch.float32, f"fp32 buffer has invalid dtype ({fp32.dtype})"

    # Undo bf16 rounding
    bf16 = bf16.view(torch.int16) - torch.where(rem < 0, 1, 0)

    # Pack bf16 and remainder into little-endian fp32
    fp32 = fp32.unsqueeze(-1).view(torch.int16)
    fp32 = torch.stack((rem, bf16), dim=-1, out=fp32)


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
        overlap_grad_sync (boolean, optional): whether to overlap
            gradient synchronization with backward pass compute
            (default: True)
        overlap_param_sync (boolean, optional): whether to overlap
            parameter synchronization with forward pass compute
            (default: False). This is an experimental feature.
        bucket_cap_mb (float, optional): bucket size in megabytes
            (default: 100)
        pipeline_size (int, optional): number of buckets to process
            simultaneously in optimizer step (default: 2)
        contiguous_param_buffer (bool, optional): convert parameters
            into views into large persistent buffers (default: False).
            This enables some performance optimizations (e.g. avoiding
            some memory copies), but may add memory overhead (e.g. if
            the memory allocator can't reuse the original parameter
            buffers).
        contiguous_grad_buffer (bool, optional): allocate gradient
            buckets out of a large persistent buffers (default:
            False). This allows individual parameter gradients to be
            accessed externally (see grad_buffer_view function). It
            enables some performance optimizations (e.g. avoiding some
            memory copies), but prevents some memory optimizations
            (e.g. the memory allocator can't reuse buffers for
            gradient buckets).
        store_params (bool, optional): store a distributed copy of the
            parameters as optimizer state (default: True). This may be
            desirable if the optimizer dtype has higher precision than
            the parameter dtype.
        store_param_remainders (bool, optional): if model is BF16 and
            optimizer is FP32, store bits required to reconstruct FP32
            params (default: False). This is an experimental feature.
        with_scaled_states (bool, optional): apply per-tensor scaling
            factors to the optimizer state (default: False). As
            discussed in `FP8-LM: Training FP8 Large Language
            Models`_, this helps maintain a reasonable dynamic range
            even when the state is in a low-precision datatype like
            FP16.
        nccl_ub (bool, optional): enable NCCL user buffers for zero-copy
            (default: False). It allows the collectives to use only 1 SM
            when IB SHARP is enabled in a one-rank-per-node communication 
            group. This will help speedup the gemms overlapped with data-
            parallel communications.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
    .. _ZeRO\: Memory Optimizations Toward Training Trillion Parameter Models:
        https://arxiv.org/abs/1910.02054
    .. _FP8-LM\: Training FP8 Large Language Models:
        https://arxiv.org/pdf/2310.18313v2.pdf

    """

    @dataclass
    class ParameterFragment:
        """Buffer ranges for a parameter fragment

        Describes corresponding regions in parameter buffer and
        parameter bucket.

        """

        # Parameter group index
        param_group_id: int
        # Parameter index within parameter group
        param_id: int
        # Bucket index
        bucket_id: int
        # Range within flattened parameter buffer
        param_range: Tuple[int, int]
        # Range within bucket
        bucket_range: Tuple[int, int]
        # Whether fragment is in local shard of bucket
        in_local_shard: bool
        # Range within local shard
        shard_range: Optional[Tuple[int, int]]
        # Range of local fragment shard within bucket
        shard_bucket_range: Optional[Tuple[int, int]]
        # Range of local fragment shard within parameter
        shard_param_range: Optional[Tuple[int, int]]

    class StateBucket:
        """Optimizer state for a bucket"""

        def __init__(
            self,
            bucket_size: int,
            shard_size: int,
            dtype: torch.dtype,
            device: torch.device,
            grad_sync_dtype: torch.dtype,
            param_sync_dtype: torch.dtype,
            contiguous_buffer_offset: int = 0,
            store_params: bool = False,
            store_param_remainders: bool = False,
        ):
            # Size of parameter bucket
            self.bucket_size: int = bucket_size
            # Size of local shard of parameter bucket
            self.shard_size: int = shard_size
            # Data type for state
            self.dtype = dtype
            # Data type for gradient synchronization
            self.grad_sync_dtype = grad_sync_dtype
            # Data type for parameter synchronization
            self.param_sync_dtype = param_sync_dtype
            # Size of the filled region in the bucket
            self.filled_size: int = 0
            # Is it able to continue filling
            self.able_to_fill: bool = True
            # Offset to bucket in contiguous buffers
            self.contiguous_buffer_offset: int = contiguous_buffer_offset
            # Buffer ranges corresponding to parameter fragments
            self.fragments: List[ParameterFragment] = []
            # Local shard of parameters
            self.params_shard: Optional[torch.Tensor] = None
            if store_params:
                self.params_shard = torch.zeros(
                    [shard_size],
                    dtype=self.dtype,
                    device=device,
                )
            # Local shard of parameter remainders
            self.param_remainders_shard: Optional[torch.Tensor] = None
            if store_param_remainders:
                self.param_remainders_shard = torch.zeros(
                    [shard_size],
                    dtype=torch.int16,
                    device=device,
                )
            # Local shard of first moment estimate
            self.exp_avg_shard: torch.Tensor = torch.zeros(
                [shard_size],
                dtype=self.dtype,
                device=device,
            )
            # Local shard of second moment estimate
            self.exp_avg_sq_shard: torch.Tensor = torch.zeros(
                [shard_size],
                dtype=self.dtype,
                device=device,
            )

        def dtypes(self) -> Tuple[torch.dtype, torch.dtype, torch.dtype]:
            """Datatypes for the bucket's compute and communication"""
            return (
                self.dtype,
                self.grad_sync_dtype,
                self.param_sync_dtype,
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
            self.grads_shard: Optional[torch.Tensor] = None
            # Local contribution to gradients
            self.grads_bucket: Optional[torch.Tensor] = None
            # Buffer for gradient reduce-scatter
            self.sync_grads_shard: Optional[torch.Tensor] = None
            # Status of gradients
            self.status: GradientStatus = DistributedFusedAdam.GradientStatus.READY
            # Params that have generated grads
            self.grads_generated: Set[torch.nn.Parameter] = set()

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
            self.params_shard: Optional[torch.Tensor] = None
            # Gathered parameter values
            self.params_bucket: Optional[torch.Tensor] = None
            # Status of parameters
            self.status: ParameterStatus = DistributedFusedAdam.ParameterStatus.SHARDED
            # Params that have been updated
            self.params_updated: Set[torch.nn.Parameter] = set()

    # Enable custom logic for AMP grad scaling
    _step_supports_amp_scaling: bool = True
    _custom_amp_unscale_grads: bool = True

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[dict]],
        lr: float = 1e-3,
        bias_correction: bool = True,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        adam_w_mode: bool = True,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        dtype: torch.dtype = torch.float32,
        grad_sync_dtype: Optional[torch.dtype] = None,
        param_sync_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = "cuda",
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        distributed_process_group: Optional[torch.distributed.ProcessGroup] = None,
        redundant_process_group: Optional[torch.distributed.ProcessGroup] = None,
        average_grad_sync: bool = True,
        overlap_grad_sync: bool = True,
        overlap_param_sync: bool = False,
        bucket_cap_mb: float = 100.0,
        pipeline_size: int = 2,
        contiguous_param_buffer: bool = False,
        contiguous_grad_buffer: bool = False,
        store_params: bool = True,
        store_param_remainders: bool = False,
        with_scaled_states: bool = False,
        nccl_ub: bool = False,
    ):
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Adam options
        self.adam_w_mode: bool = adam_w_mode
        if amsgrad:
            raise RuntimeError(
                "DistributedFusedAdam does not support the AMSGrad variant."
            )

        # Datatype options
        if grad_sync_dtype is None:
            grad_sync_dtype = dtype
        if param_sync_dtype is None:
            param_sync_dtype = dtype
        supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if (
            dtype not in supported_dtypes
            or grad_sync_dtype not in supported_dtypes
        ):
            raise ValueError(
                "Unsupported dtypes for DistributedFusedAdam "
                f"(dtype={dtype}, "
                f"grad_sync_dtype={grad_sync_dtype}, "
                f"param_sync_dtype={param_sync_dtype}))"
            )
        self.dtype: torch.dtype = dtype
        self.grad_sync_dtype: torch.dtype = grad_sync_dtype
        self.param_sync_dtype: torch.dtype = param_sync_dtype

        # Device options
        if not _devices_match(device, "cuda"):
            raise RuntimeError(
                "Invalid device for DistributedFusedAdam " f"(device={device})"
            )
        self.device: torch.device = torch.device("cuda", torch.cuda.current_device())

        # Process groups
        self.process_group: torch.distributed.ProcessGroup = (
            _get_default_group() if process_group is None else process_group
        )
        self.distributed_process_group: torch.distributed.ProcessGroup = (
            self.process_group
            if distributed_process_group is None
            else distributed_process_group
        )
        self.redundant_process_group: Optional[
            torch.distributed.ProcessGroup
        ] = redundant_process_group
        self.process_group_size: int = torch.distributed.get_world_size(
            self.process_group
        )
        self.distributed_rank: int = torch.distributed.get_rank(
            self.distributed_process_group
        )
        self.distributed_size: int = torch.distributed.get_world_size(
            self.distributed_process_group
        )
        self.redundant_size: int = (
            1
            if self.redundant_process_group is None
            else torch.distributed.get_world_size(self.redundant_process_group)
        )
        if self.process_group_size != self.distributed_size * self.redundant_size:
            raise RuntimeError(
                "Invalid process group configuration "
                f"(process group size = {self.process_group_size}, "
                f"distributed process group size = {self.distributed_size}, "
                f"redundant process group size = {self.redundant_size})"
            )
        self.process_group_root: int = get_global_rank(self.process_group, 0)

        # Use average reduction for grad sync
        self.average_grad_sync: bool = average_grad_sync
        # Copy param grads to bucket as soon as available
        self.greedy_grad_copy: bool = True
        # Synchronize grad buckets as soon as their grads are available
        self.overlap_grad_sync: bool = overlap_grad_sync
        # Try synchronizing param buckets just before param is needed
        self.overlap_param_sync: bool = overlap_param_sync
        # Number of buckets to synchronize at a time
        self.pipeline_size: int = pipeline_size

        # Store params or param remainders
        if store_param_remainders:
            if store_params:
                raise RuntimeError(
                    "Attempted to construct DistributedFusedAdam "
                    "with store_params=True and store_param_remainders=True"
                )
            if self.dtype != torch.float32 or self.param_sync_dtype != torch.bfloat16:
                raise RuntimeError(
                    "DistributedFusedAdam requires "
                    "BF16 params and FP32 optimizer state "
                    "when storing parameter remainders "
                    f"(dtype={self.dtype}, "
                    f"param_sync_dtype={self.param_sync_dtype}))"
                )
        self.store_params: bool = store_params
        self.store_param_remainders: bool = store_param_remainders

        # Whether to scale optimizer state
        self.with_scaled_states: bool = with_scaled_states
        if self.with_scaled_states:
            if not self.store_params:
                raise RuntimeError(
                    "Attempted to construct DistributedFusedAdam "
                    "with with_scaled_state=True and store_params=False"
                )
            if self.store_param_remainders:
                raise RuntimeError(
                    "Attempted to construct DistributedFusedAdam "
                    "with with_scaled_state=True and store_params_remainders=True"
                )
            if self.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    "Attempted to construct DistributedFusedAdam "
                    f"with with_scaled_state=True and dtype={self.dtype} "
                    "(only fp16 and bf16 are supported)"
                )
            if self.param_sync_dtype == torch.float32:
                # _local_step_with_scaled_states applies Adam kernel
                # to fp32 workspace buffer and relies on
                # _check_params_shard_dtypes to copy to param sync
                # workspace buffer. However,
                # _check_params_shard_dtypes does nothing if
                # param_sync_dtype is fp32.
                raise RuntimeError(
                    "Attempted to construct DistributedFusedAdam "
                    f"with with_scaled_state=True and param_sync_dtype={self.param_sync_dtype}"
                )
        # Scaling factors to apply to recover unscaled optimizer state
        self._state_scales: dict = {}

        # Determine bucket sizes
        dtype_size = torch.finfo(self.grad_sync_dtype).bits // 8
        self.alignment: int = 128 // dtype_size
        bucket_size = 1024 * 1024 * bucket_cap_mb / dtype_size
        shard_size = int(bucket_size / self.distributed_size)
        shard_size = _round_to_multiple(shard_size, self.alignment, round_up=False)
        shard_size = max(shard_size, self.alignment)
        self.default_shard_size: int = shard_size

        # Optimizer state
        self.state["buckets"]: List[StateBucket] = []
        self.state["step"]: int = 0

        # Gradient state
        self._grads_buckets: Dict[int, GradientBucket] = collections.defaultdict(
            self.GradientBucket
        )
        # Param state
        self._params_buckets: Dict[int, ParameterBucket] = collections.OrderedDict()

        # Whether to allocate contiguous buffers for parameters
        self.contiguous_param_buffer: bool = contiguous_param_buffer
        # Whether to allocate contiguous buffers for gradients
        self.contiguous_grad_buffer: bool = contiguous_grad_buffer
        # Whether to use NCCL User Buffer
        self.nccl_ub: bool = nccl_ub
        # Contiguous buffers for parameters
        self._param_buffers: Dict[
            Tuple[torch.dtype, torch.dtype, torch.dtype], torch.Tensor
        ] = {}
        # Contiguous buffers for gradients
        self._grad_buffers: Dict[
            Tuple[torch.dtype, torch.dtype, torch.dtype], torch.Tensor
        ] = {}
        # Output buffer for gradient shards, only required for NCCL user buffer
        if self.nccl_ub: 
            if not nccl_allocator:
                raise RuntimeError("NCCL allocator importing failed but nccl ub is still requested")
            elif not self.contiguous_grad_buffer:
                raise RuntimeError("NCCL user buffers require contiguous grad buffers")
            else:
                self._shard_grad_buffers: Dict[
                    Tuple[torch.dtype, torch.dtype, torch.dtype], torch.Tensor
                ] = {}

        # Side streams for optimizer step and communication
        self._pipeline_streams: List[torch.cuda.Stream] = [
            torch.cuda.Stream() for _ in range(self.pipeline_size + 1)
        ]

        # Scale by factor before optimizer step. Used for grad
        # clipping and gradient scaler.
        self._grad_scale: torch.Tensor = torch.full(
            [], 1.0, dtype=torch.float32, device=self.device
        )
        # Norm of parameter gradients. Used for gradient clipping and
        # gradient scaler.
        self._grad_norm: Optional[torch.Tensor] = None

        # Dummy flag for multi-tensor kernels
        # Note: Apex multi-tensor kernels have a noop_flag argument
        # that is intended to detect non-finite values. It shouldn't
        # have any effect with the kernels used in the optimizer, but
        # we still set it to zero out of an abundance of caution.
        self._dummy_overflow_buf: torch.Tensor = torch.zeros(
            [1], dtype=torch.int32, device=self.device
        )

        # Check if collectives have no_copy option
        self._gather_no_copy: bool = (
            "no_copy" in inspect.getfullargspec(torch.distributed.gather).args
        )

        # Make sure parameter values are same across processes
        self._broadcast_params()

        # Lock for callbacks
        self._lock: threading.Lock = threading.Lock()
        # Attach hooks for gradient synchronization
        self._register_post_backward_hooks()
        # Attach hooks for param synchronization
        if self.overlap_param_sync:
            self._register_pre_forward_hooks()

    @torch.no_grad()
    def _broadcast_params(self) -> None:
        """Broadcast parameter values from root rank"""
        process_group = self.process_group
        with _coalescing_manager(process_group, self.device, async_ops=True) as cm:
            for param_group in self.param_groups:
                for param in param_group["params"]:
                    _coalescing_manager_append_work(
                        cm,
                        torch.distributed.broadcast(
                            param,
                            src=self.process_group_root,
                            group=process_group,
                            async_op=True,
                        ),
                    )
        cm.wait()

    def _make_post_backward_hook(
        self,
        param: torch.nn.Parameter,
        param_group_id: int,
        param_id: int,
    ) -> Callable:
        """Create callback function to call after param generates grad

        Lazily initialize parameter and try launching grad sync.

        """

        def post_backward_hook(*unused) -> None:
            if getattr(param, "_pre_forward_hook_is_enabled", False):
                raise RuntimeError(
                    "A parameter called its post-backward hook "
                    "before its pre-forward hook. "
                    "Please manually interact with the parameter "
                    "before the forward pass (e.g. by calling data_ptr) "
                    "or run DistributedFusedAdam with overlap_param_sync=False."
                )
            with self._lock:
                need_to_initialize = "fragments" not in self.state[param]
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

    def _register_post_backward_hooks(self) -> None:
        """Attach hooks for gradient synchronization"""
        self._grad_accs = []
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group["params"]):
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

    def _make_pre_forward_hook(
        self,
        param: torch.nn.Parameter,
        param_group_id: int,
        param_id: int,
    ) -> Callable:
        """Create callback function to call before param forward pass

        Make sure param has been synchronized and try launching next
        param sync.

        """

        def pre_forward_hook(*unused) -> None:
            with self._lock:
                if "fragments" not in self.state[param]:
                    return
                self._param_copy(param)
                if self.overlap_param_sync:
                    self._try_start_bucket_param_sync()

        return pre_forward_hook

    def _register_pre_forward_hooks(self) -> None:
        """Attach hooks for parameter synchronization

        If _pre_forward_hook_is_enabled is set in a parameter, then
        the callback will be called the first time any of its
        attributes are accessed. This is hackily done by
        monkey-patching the parameter class, so proceed with caution.

        """
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group["params"]):
                # Monkey-patch parameter class
                cls = param.__class__
                if not getattr(cls, "_has_pre_forward_hook", False):
                    # Monkey-patch magic methods to call __getattribute__
                    special_funcs = [
                        "__abs__",
                        "__add__",
                        "__and__",
                        "__bool__",
                        "__complex__",
                        "__contains__",
                        "__deepcopy__",
                        "__delitem__",
                        "__div__",
                        "__eq__",
                        "__float__",
                        "__floordiv__",
                        "__ge__",
                        "__getitem__",
                        "__gt__",
                        "__iadd__",
                        "__iand__",
                        "__idiv__",
                        "__ifloordiv__",
                        "__ilshift__",
                        "__imod__",
                        "__imul__",
                        "__index__",
                        "__int__",
                        "__invert__",
                        "__ior__",
                        "__ipow__",
                        "__irshift__",
                        "__isub__",
                        "__iter__",
                        "__itruediv__",
                        "__ixor__",
                        "__le__",
                        "__len__",
                        "__long__",
                        "__lshift__",
                        "__lt__",
                        "__matmul__",
                        "__mod__",
                        "__mul__",
                        "__neg__",
                        "__nonzero__",
                        "__or__",
                        "__pos__",
                        "__pow__",
                        "__radd__",
                        "__rand__",
                        "__rdiv__",
                        "__reduce__",
                        "__reduce_ex__",
                        "__reversed__",
                        "__rfloordiv__",
                        "__rlshift__",
                        "__rmatmul__",
                        "__rmod__",
                        "__rmul__",
                        "__ror__",
                        "__rpow__",
                        "__rrshift__",
                        "__rshift__",
                        "__rsub__",
                        "__rtruediv__",
                        "__rxor__",
                        "__setitem__",
                        "__sizeof__",
                        "__sub__",
                        "__torch_function__",
                        "__truediv__",
                        "__xor__",
                    ]
                    for func_name in special_funcs:

                        def make_augmented_func() -> Callable:
                            base_func_name = f"_base_{func_name}"

                            def augmented_func(self, *args, **kwargs):
                                return getattr(self, base_func_name)(*args, **kwargs)

                            return augmented_func

                        setattr(cls, f"_base_{func_name}", getattr(cls, func_name))
                        setattr(cls, func_name, make_augmented_func())

                    # Monkey-patch __getattribute__ to call pre-forward hook
                    def make_getattribute() -> Callable[[str], Any]:
                        special_attrs = {
                            "_pre_forward_hook_is_enabled",
                            "_pre_forward_hook",
                            "__del__",
                            "__delattr__",
                            "__dir__",
                            "__getattr__",
                            "__getattribute__",
                            "__hash__",
                            "__init__",
                            "__new__",
                            "__setattr__",
                        }

                        def getattribute_with_pre_forward_hook(self, name: str):
                            """Variant of __getattribute__ that can call pre-forward hook"""
                            if name not in special_attrs:
                                if getattr(self, "_pre_forward_hook_is_enabled", False):
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

    @torch.no_grad()
    def init_param_buffer(self) -> None:
        """Allocate contiguous buffers for param buckets

        This converts the parameters into views into contiguous
        buffers. This enables some performance optimizations (e.g.
        avoiding some memory copies), but may add memory overhead
        (e.g. if the memory allocator can't reuse the original
        parameter buffers). To minimize memory overhead, this buffer
        should be initialized before the first training step.

        """

        # Make sure all params are initialized
        self.contiguous_param_buffer = True
        self.init_params()

        # Construct param buffers
        buffer_sizes = collections.defaultdict(lambda: 0)
        for bucket in self.state["buckets"]:
            dtypes = bucket.dtypes()
            buffer_sizes[dtypes] = max(
                bucket.contiguous_buffer_offset + bucket.bucket_size,
                buffer_sizes[dtypes],
            )
        for dtypes, buffer_size in buffer_sizes.items():
            _, _, param_sync_dtype = dtypes
            self._param_buffers[dtypes] = torch.zeros(
                [buffer_size],
                dtype=param_sync_dtype,
                device=self.device,
            )

        # Figure out corresponding positions in params and param buffer
        params = list(self.parameters())
        param_flat_views = []
        param_buffer_views = []
        for i, param in enumerate(params):
            fragment = self.state[param]["fragments"][0]
            bucket_id = fragment.bucket_id
            bucket = self.state["buckets"][bucket_id]
            param_size = param.numel()
            bucket_start, _ = fragment.bucket_range
            buffer_offset = bucket.contiguous_buffer_offset
            buffer_start = buffer_offset + bucket_start
            buffer_end = buffer_start + param_size
            param_buffer = self._param_buffers[bucket.dtypes()]
            param_buffer_view = param_buffer[buffer_start:buffer_end].detach()
            if not _devices_match(param_buffer_view.device, param.device):
                raise RuntimeError(
                    "Attempted to change a parameter with device={param.device} "
                    f"into a buffer view with device={param_buffer_view.device}"
                )
            if param_buffer_view.dtype != param.dtype:
                if (
                    not torch.is_floating_point(param_buffer_view)
                    and param_buffer_view.element_size() == param.element_size()
                ):
                    param_buffer_view = param_buffer_view.view(dtype=param.dtype)
                else:
                    raise RuntimeError(
                        f"Attempted to change a parameter with dtype={param.dtype} "
                        f"into a buffer view with dtype={param_buffer_view.dtype}"
                    )
            param_flat_views.append(param.detach().view(-1))
            param_buffer_views.append(param_buffer_view)

        # Copy values into param buffer
        _multi_tensor_copy(
            param_flat_views,
            param_buffer_views,
            dummy_overflow_buf=self._dummy_overflow_buf,
        )

        # Make all params a view into the param buffer
        for param, buffer_view in zip(params, param_buffer_views):
            param.data = buffer_view.view(param.size())

    def _init_grad_buffer(self) -> None:
        """Allocate contiguous buffer for grad buckets"""

        # Make sure all params are initialized
        self.contiguous_grad_buffer = True
        self.init_params()

        # Construct grad buffers
        buffer_sizes = collections.defaultdict(lambda: 0)
        for bucket in self.state["buckets"]:
            dtypes = bucket.dtypes()
            buffer_sizes[dtypes] = max(
                bucket.contiguous_buffer_offset + bucket.bucket_size,
                buffer_sizes[dtypes],
            )
        for dtypes, buffer_size in buffer_sizes.items():
            _, grad_sync_dtype, _ = dtypes
            if not self.nccl_ub:
                self._grad_buffers[dtypes] = torch.zeros(
                    [buffer_size], dtype=grad_sync_dtype, device=self.device,
                )
            else:
                with nccl_allocator.nccl_mem():
                    self._grad_buffers[dtypes] = torch.zeros(
                        [buffer_size], dtype=grad_sync_dtype, device=self.device,
                    )
                shard_buffer_size = buffer_size // self.distributed_size
                with nccl_allocator.nccl_mem():
                    self._shard_grad_buffers[dtypes] = torch.zeros(
                        [shard_buffer_size], dtype=grad_sync_dtype, device=self.device,
                    )

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """Returns an iterator over optimizer parameters"""
        return itertools.chain.from_iterable(
            group["params"] for group in self.param_groups
        )

    def parameter(
        self,
        *args: Union[int, ParameterFragment],
    ) -> torch.nn.Parameter:
        """Get optimizer parameter

        Can either accept two ints or one
        DistributedFusedAdam.ParameterFragment.

        Arguments:
            param_group_id (int): Parameter group index
            param_id (int): Parameter index within parameter group

        """
        if (
            len(args) == 2
            and isinstance(args[0], int)
            and isinstance(args[1], int)
        ):
            param_group_id = args[0]
            param_id = args[1]
        elif len(args) == 1 and isinstance(args[0], self.ParameterFragment):
            fragment = args[0]
            param_group_id = fragment.param_group_id
            param_id = fragment.param_id
        else:
            raise TypeError(
                "Expected input types are "
                "[int, int] or [DistributedFusedAdam.ParameterFragment], "
                f"but found {[type(arg).__name__ for arg in args]}"
            )
        return self.param_groups[param_group_id]["params"][param_id]

    def init_params(
        self,
        params: Optional[Iterable[torch.nn.Parameter]] = None,
        dtype: Optional[torch.dtype] = None,
        grad_sync_dtype: Optional[torch.dtype] = None,
        param_sync_dtype: Optional[torch.dtype] = None,
    ) -> None:
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
        params = [param for param in params if "fragments" not in self.state[param]]
        if not params:
            return

        # Get indices corresponding to parameters
        id_map = dict()
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group["params"]):
                id_map[param] = (param_group_id, param_id)

        # Initialize parameters
        for param in params:
            if param in id_map:
                param_group_id, param_id = id_map[param]
                self._init_param_state(
                    param,
                    param_group_id,
                    param_id,
                    dtype=dtype,
                    grad_sync_dtype=grad_sync_dtype,
                    param_sync_dtype=param_sync_dtype,
                )

    def init_params_bucket(
        self,
        params: Iterable[torch.nn.Parameter],
        dtype: Optional[torch.dtype] = None,
        grad_sync_dtype: Optional[torch.dtype] = None,
        param_sync_dtype: Optional[torch.dtype] = None,
    ) -> None:
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
        params = [param for param in params if "fragments" not in self.state[param]]
        if not params:
            return

        # Get indices corresponding to parameters
        id_map = dict()
        for param_group_id, group in enumerate(self.param_groups):
            for param_id, param in enumerate(group["params"]):
                id_map[param] = [param_group_id, param_id]
        param_ids = [tuple([param] + id_map[param]) for param in params]

        # Mark existings bucket as fully filled
        for bucket in self.state["buckets"]:
            bucket.able_to_fill = False

        # Initialize optimizer state for parameters
        start_bucket_id = len(self.state["buckets"])
        self.init_params(
            params,
            dtype=dtype,
            grad_sync_dtype=grad_sync_dtype,
            param_sync_dtype=param_sync_dtype,
        )
        end_bucket_id = len(self.state["buckets"])

        # Make sure all added buckets depend on provided params
        for bucket_id in range(start_bucket_id, end_bucket_id):
            bucket = self.state["buckets"][bucket_id]
            bucket_size = bucket.bucket_size
            bucket.able_to_fill = False
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
                        shard_range=None,
                        shard_bucket_range=None,
                        shard_param_range=None,
                    )
                    self.state[param]["fragments"].append(fragment)
                    bucket.fragments.append(fragment)

    @torch.no_grad()
    def _init_param_state(
        self,
        param: torch.nn.Parameter,
        param_group_id: int,
        param_id: int,
        dtype: Optional[torch.dtype] = None,
        grad_sync_dtype: Optional[torch.dtype] = None,
        param_sync_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize optimizer state for a parameter"""

        # Return immediately if already initialized
        if "fragments" in self.state[param]:
            return
        self.state[param]["fragments"] = []

        # Data type configuration
        if dtype is None:
            dtype = self.dtype
        if grad_sync_dtype is None:
            grad_sync_dtype = self.grad_sync_dtype
        if param_sync_dtype is None:
            param_sync_dtype = self.param_sync_dtype
        if dtype != self.dtype:
            raise ValueError(
                "Optimizer states with non-default dtypes are not supported"
            )
        supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if (
            dtype not in supported_dtypes
            or grad_sync_dtype not in supported_dtypes
        ):
            raise ValueError(
                "Unsupported dtypes for DistributedFusedAdam "
                f"(dtype={dtype}, "
                f"grad_sync_dtype={grad_sync_dtype}, "
                f"param_sync_dtype={param_sync_dtype}))"
            )

        # Store params or param remainders
        store_params = (
            self.store_params
            or dtype != self.dtype
            or param_sync_dtype != self.param_sync_dtype
        )
        store_param_remainders = (
            self.store_param_remainders
            and dtype == self.dtype
            and param_sync_dtype == self.param_sync_dtype
        )

        def last_bucket_id() -> int:
            """Index of last optimizer state bucket with desired dtypes

            -1 if there are no such buckets.

            """
            dtypes = (dtype, grad_sync_dtype, param_sync_dtype)
            bucket_id = len(self.state["buckets"]) - 1
            while bucket_id > 0:
                bucket = self.state["buckets"][bucket_id]
                if bucket.dtypes() == dtypes:
                    break
                bucket_id -= 1
            return bucket_id

        def make_bucket(
            bucket_size: int,
            shard_size: int,
            buffer_offset: int,
        ) -> None:
            """Construct new optimizer state bucket"""
            self.state["buckets"].append(
                self.StateBucket(
                    bucket_size,
                    shard_size,
                    dtype,
                    self.device,
                    grad_sync_dtype,
                    param_sync_dtype,
                    contiguous_buffer_offset=buffer_offset,
                    store_params=store_params,
                    store_param_remainders=store_param_remainders,
                )
            )

        # Make sure there is at least one bucket with expected dtypes
        if last_bucket_id() < 0:
            shard_size = self.default_shard_size
            bucket_size = shard_size * self.distributed_size
            buffer_offset = 0
            make_bucket(bucket_size, shard_size, buffer_offset)

        # Split parameter values into fragments
        # Note: Each fragment resides within a bucket
        param_start = 0
        param_size = param.numel()
        while param_start < param_size:
            # Get current bucket
            bucket_id = last_bucket_id()
            bucket = self.state["buckets"][bucket_id]
            fragment_id = len(bucket.fragments)
            bucket_size = bucket.bucket_size
            shard_size = bucket.shard_size

            # Determine fragment position within bucket
            bucket_start = _round_to_multiple(
                bucket.filled_size,
                self.alignment,
                round_up=True,
            )
            fragment_size = min(param_size - param_start, bucket_size - bucket_start)
            param_end = param_start + fragment_size
            bucket_end = bucket_start + fragment_size

            # Create new bucket if current one is full
            if fragment_size <= 0 or not bucket.able_to_fill:
                shard_size = self.default_shard_size
                bucket_size = shard_size * self.distributed_size
                buffer_offset = bucket.contiguous_buffer_offset + bucket.bucket_size
                make_bucket(bucket_size, shard_size, buffer_offset)
                continue

            # Fragment position within local shard
            shard_id = self.distributed_rank
            shard_start = bucket_start - shard_size * shard_id
            shard_end = bucket_end - shard_size * shard_id
            shard_start = min(max(shard_start, 0), shard_size)
            shard_end = min(max(shard_end, 0), shard_size)
            in_local_shard = shard_start < shard_end
            shard_range = None
            shard_bucket_range = None
            shard_param_range = None
            if in_local_shard:
                shard_range = (shard_start, shard_end)
                shard_bucket_start = shard_start + shard_size * shard_id
                shard_bucket_end = shard_bucket_start + shard_end - shard_start
                shard_bucket_range = (shard_bucket_start, shard_bucket_end)
                shard_param_start = shard_bucket_start - bucket_start + param_start
                shard_param_end = shard_param_start + shard_end - shard_start
                shard_param_range = (shard_param_start, shard_param_end)

            # Record fragment info
            fragment = self.ParameterFragment(
                param_group_id=param_group_id,
                param_id=param_id,
                bucket_id=bucket_id,
                param_range=(param_start, param_end),
                bucket_range=(bucket_start, bucket_end),
                in_local_shard=in_local_shard,
                shard_range=shard_range,
                shard_bucket_range=shard_bucket_range,
                shard_param_range=shard_param_range,
            )
            self.state[param]["fragments"].append(fragment)
            bucket.fragments.append(fragment)
            bucket.filled_size = bucket_end
            param_start = param_end

        # Initialize optimizer state scaling factors if needed
        if self.with_scaled_states:
            for fragment in self.state[param]["fragments"]:
                if not fragment.in_local_shard:
                    continue
                bucket_id = fragment.bucket_id
                self._state_scales[(param_group_id, param_id, bucket_id)] = dict(
                    param=torch.zeros([1], dtype=torch.float32, device=self.device),
                    exp_avg=torch.zeros([1], dtype=torch.float32, device=self.device),
                    exp_avg_sq=torch.zeros([1], dtype=torch.float32, device=self.device),
                )

        # Initialize main param buffer
        if store_params:
            for fragment in self.state[param]["fragments"]:
                if not fragment.in_local_shard:
                    continue
                bucket_id = fragment.bucket_id
                bucket = self.state["buckets"][bucket_id]
                param_range = slice(*fragment.shard_param_range)
                shard_range = slice(*fragment.shard_range)
                model_param_fragment = param.detach().view(-1)[param_range]
                if self.with_scaled_states:
                    model_param_fragment = torch.empty_like(
                        model_param_fragment,
                        dtype=torch.float32,
                    ).copy_(model_param_fragment)
                    self._apply_state_scale(
                        model_param_fragment,
                        self._state_scales[(param_group_id, param_id, bucket_id)]["param"],
                    )
                main_param_fragment = bucket.params_shard[shard_range]
                main_param_fragment.copy_(model_param_fragment)

        # Check if buckets are underutilized
        if all("fragments" in self.state[param] for param in self.parameters()):
            bucket_size = sum(bucket.bucket_size for bucket in self.state["buckets"])
            filled_size = sum(bucket.filled_size for bucket in self.state["buckets"])
            buckets_utilization = filled_size / bucket_size
            if buckets_utilization < 0.7:
                warnings.warn(
                    f"Only {buckets_utilization:.1%} of buckets are used. "
                    "Consider decreasing the bucket_cap_mb argument."
                )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clear parameter gradients"""

        # Reset bucket buffers
        self._grads_buckets.clear()

        # Construct views into contiguous grad buffer, if needed
        if self.contiguous_grad_buffer:
            if not self._grad_buffers:
                self._init_grad_buffer()
            for grad_buffer in self._grad_buffers.values():
                grad_buffer.zero_()
            for bucket_id, bucket in enumerate(self.state["buckets"]):
                bucket_size = bucket.bucket_size
                buffer_start = bucket.contiguous_buffer_offset
                buffer_end = buffer_start + bucket_size
                grad_buffer = self._grad_buffers[bucket.dtypes()]
                self._grads_buckets[bucket_id].grads_bucket = grad_buffer[
                    buffer_start:buffer_end
                ]
                if self.nccl_ub:
                    shard_size = bucket.shard_size
                    shard_buffer_start = (
                        bucket.contiguous_buffer_offset // self.distributed_size
                    )
                    shard_buffer_end = shard_buffer_start + shard_size
                    shard_grad_buffer = self._shard_grad_buffers[bucket.dtypes()]
                    self._grads_buckets[bucket_id].sync_grads_shard = shard_grad_buffer[
                        shard_buffer_start:shard_buffer_end
                    ]

        # Reset param grads
        for param in self.parameters():
            with _disable_pre_forward_hook(param):
                need_to_zero = True
                if set_to_none:
                    param.grad = None
                elif self.contiguous_grad_buffer:
                    bucket_id = self.state[param]["fragments"][0].bucket_id
                    bucket = self.state["buckets"][bucket_id]
                    if param.dtype == bucket.grad_sync_dtype and _devices_match(
                        param.device, self.device
                    ):
                        param.grad = self.grad_buffer_view(param)
                        need_to_zero = False
                if need_to_zero and param.grad is not None:
                    param.grad.zero_()

        # Reset other state
        self._grad_scale = torch.full([], 1.0, dtype=torch.float32, device=self.device)
        self._grad_norm = None
        self._dummy_overflow_buf = torch.zeros(
            [1], dtype=torch.int32, device=self.device
        )

    def _grad_copy(self, param: torch.nn.Parameter) -> None:
        """Copy parameter gradients to gradient buckets

        Initializes gradient buckets if needed. The original parameter
        gradient is set to None.

        """

        # Initialize parameter if needed
        if "fragments" not in self.state[param]:
            for param_group_id, group in enumerate(self.param_groups):
                for param_id, param_ in enumerate(group["params"]):
                    if param is param_:
                        self._init_param_state(param, param_group_id, param_id)
            if "fragments" not in self.state[param]:
                raise RuntimeError(
                    "Could not initialize DistributedFusedAdam with parameter"
                )

        # Copy param grad to buckets
        for fragment in self.state[param]["fragments"]:
            # Get fragment position
            bucket_id = fragment.bucket_id
            bucket = self._grads_buckets[bucket_id]
            bucket_size = self.state["buckets"][bucket_id].bucket_size
            grad_sync_dtype = self.state["buckets"][bucket_id].grad_sync_dtype
            grad_start, grad_end = fragment.param_range
            bucket_start, bucket_end = fragment.bucket_range

            # Set reduction status
            if bucket.status == self.GradientStatus.SYNCING:
                self._finish_bucket_grad_sync()
            bucket.status = self.GradientStatus.PARTIALLY_FILLED

            # Allocate gradient buffer if needed
            if bucket.grads_bucket is None and self.contiguous_grad_buffer:
                if not self._grad_buffers:
                    self._init_grad_buffer()
                state_bucket = self.state["buckets"][bucket_id]
                buffer_start = state_bucket.contiguous_buffer_offset
                buffer_end = buffer_start + bucket_size
                grad_buffer = self._grad_buffers[state_bucket.dtypes()]
                grad_buffer = grad_buffer[buffer_start:buffer_end]
                if (
                    bucket.grads_shard is None
                    or bucket.grads_shard.storage().data_ptr()
                    != grad_buffer.storage().data_ptr()
                ):
                    bucket.grads_bucket = grad_buffer
                    bucket.grads_bucket.zero_()
            if bucket.grads_bucket is None:
                bucket.grads_bucket = torch.zeros(
                    [bucket_size],
                    dtype=grad_sync_dtype,
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

    def _param_copy(
        self,
        params: Union[torch.nn.Parameter, Iterable[torch.nn.Parameter]],
    ) -> None:
        """Update parameters with values from parameter buckets

        Synchronizes and deletes parameter buckets as needed.

        """

        # Get parameter fragments to be synchronized
        if isinstance(params, torch.Tensor):
            params = [params]
        fragments = []
        for param in params:
            if "fragments" in self.state[param]:
                fragments.extend(
                    fragment
                    for fragment in self.state[param]["fragments"]
                    if fragment.bucket_id in self._params_buckets
                )

        # Return immediately if no fragments need to be synchronized
        if not fragments:
            return

        # Make sure all needed buckets have been synchronized
        buckets = collections.OrderedDict()
        for fragment in fragments:
            bucket_id = fragment.bucket_id
            bucket = self._params_buckets[bucket_id]
            buckets[bucket] = bucket.status
        if any(
            status != self.ParameterStatus.READY for bucket, status in buckets.items()
        ):
            self._start_bucket_param_sync(buckets.keys())
            self._finish_bucket_param_sync()

        # Copy values from bucket buffers to params
        self._param_copy_fragments(fragments)

        # Delete buckets if possible
        for fragment in fragments:
            bucket_id = fragment.bucket_id
            bucket = self._params_buckets[bucket_id]
            bucket.params_updated.add(self.parameter(fragment))
            bucket_fragments = self.state["buckets"][bucket_id].fragments
            if len(bucket.params_updated) == len(bucket_fragments):
                del self._params_buckets[bucket_id]

    def _param_copy_fragments(
        self,
        fragments: Iterable[ParameterFragment],
    ) -> None:
        """Update parameter fragments with values from parameter buckets"""

        # Figure out corresponding positions in param buckets and params
        buffers_in = []
        buffers_out = []
        for fragment in fragments:

            # Check if fragment needs to be updated
            bucket_id = fragment.bucket_id
            bucket_start, bucket_end = fragment.bucket_range
            param_start, param_end = fragment.param_range
            if param_end <= param_start or bucket_id not in self._params_buckets:
                continue

            # Corresponding positions in param bucket and param
            bucket = self._params_buckets[bucket_id]
            param = self.parameter(fragment)
            buffer_in = bucket.params_bucket[bucket_start:bucket_end]
            buffer_out = param.detach().view(-1)[param_start:param_end]

            if (
                torch.is_floating_point(buffer_in)
                and torch.is_floating_point(buffer_out)
            ):
                # Cast between floating-point dtypes
                buffers_in.append(buffer_in)
                buffers_out.append(buffer_out)
            else:
                # Copy most significant bytes for non-floating-point
                # dtypes
                # Note: Assume dtypes are little-endian
                in_bytes = buffer_in.unsqueeze(-1).view(torch.uint8)
                out_bytes = buffer_out.unsqueeze(-1).view(torch.uint8)
                copy_size = min(in_bytes.size(-1), out_bytes.size(-1))
                buffers_in.append(in_bytes[..., -copy_size:])
                buffers_out.append(out_bytes[..., -copy_size:])
                if copy_size < out_bytes.size(-1):
                    out_bytes[..., :-copy_size].zero_()

        # Copy data from parameter buckets to parameters
        _multi_tensor_copy(
            buffers_in,
            buffers_out,
            dummy_overflow_buf=self._dummy_overflow_buf,
        )

    def grad_buffer_view(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Construct view into grad buffer corresponding to param

        Assumes optimizer is using a contiguous grad buffer.

        """

        # Initialize contiguous grad buffers if needed
        assert self.contiguous_grad_buffer
        if not self._grad_buffers:
            self._init_grad_buffer()

        # Figure out corresponding position in grad buffer
        fragment = self.state[param]["fragments"][0]
        bucket_id = fragment.bucket_id
        bucket = self.state["buckets"][bucket_id]
        bucket_start, _ = fragment.bucket_range
        buffer_offset = bucket.contiguous_buffer_offset
        buffer_start = buffer_offset + bucket_start
        buffer_end = buffer_start + param.numel()

        # Construct view into grad buffer
        flat_buffer = self._grad_buffers[bucket.dtypes()]
        flat_buffer = flat_buffer[buffer_start:buffer_end]
        return flat_buffer.detach().view(param.size())

    def _force_bucket_grad_sync(self) -> None:
        """Ensure that all gradient buckets are synchronized"""

        # Synchronize all unsynchronized buckets
        Status = self.GradientStatus
        buckets = []
        for bucket_id, grads_bucket in sorted(self._grads_buckets.items()):
            if grads_bucket.status not in (Status.READY, Status.SYNCING):
                buckets.append(grads_bucket)
                if grads_bucket.grads_bucket is None:
                    state_bucket = self.state["buckets"][bucket_id]
                    grads_bucket.grads_bucket = torch.zeros(
                        [state_bucket.bucket_size],
                        dtype=state_bucket.grad_sync_dtype,
                        device=self.device,
                    )
        if buckets:
            self._start_bucket_grad_sync(buckets)
        self._finish_bucket_grad_sync()

        # Fill any unsynchronized gradients with zeros
        for bucket_id in range(len(self.state["buckets"])):
            grads_bucket = self._grads_buckets[bucket_id]
            if grads_bucket.grads_shard is None:
                state_bucket = self.state["buckets"][bucket_id]
                grads_bucket.grads_shard = torch.zeros(
                    [state_bucket.shard_size],
                    dtype=state_bucket.grad_sync_dtype,
                    device=self.device,
                )

    def _try_start_bucket_grad_sync(
        self,
        params: Optional[Iterable[torch.nn.Parameter]] = None,
        ignore_last_bucket: bool = False,
    ) -> None:
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
        if params is None:
            params = []
        for param in params:
            for fragment in self.state[param]["fragments"]:
                bucket_id = fragment.bucket_id
                grads_bucket = self._grads_buckets[bucket_id]
                state_bucket = self.state["buckets"][bucket_id]
                bucket_fragments = state_bucket.fragments
                grads_bucket.grads_generated.add(param)
                if len(grads_bucket.grads_generated) == len(bucket_fragments):
                    grads_bucket.status = self.GradientStatus.FULLY_FILLED
                    if grads_bucket.grads_bucket is None:
                        grads_bucket.grads_bucket = torch.zeros(
                            [state_bucket.bucket_size],
                            dtype=state_bucket.grad_sync_dtype,
                            device=self.device,
                        )

        # Launch reductions if enough buckets are ready
        filled_buckets = []
        for bucket_id, bucket in sorted(self._grads_buckets.items()):
            if ignore_last_bucket and bucket_id == len(self.state["buckets"]) - 1:
                continue
            if bucket.status == self.GradientStatus.FULLY_FILLED:
                filled_buckets.append(bucket)
        if filled_buckets:
            self._start_bucket_grad_sync(filled_buckets)

    def _start_bucket_grad_sync(self, buckets: List[GradientBucket]) -> None:
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
        if self.average_grad_sync and not self.nccl_ub:
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
            elif bucket.sync_grads_shard is None:
                bucket_size = bucket.grads_bucket.numel()
                shard_size = bucket_size // self.distributed_size
                bucket.sync_grads_shard = torch.empty(
                    [shard_size],
                    dtype=bucket.grads_bucket.dtype,
                    device=bucket.grads_bucket.device,
                )

            # Handle case with multiple grad accumulation steps
            if bucket.grads_shard is not None:
                if bucket.sync_grads_shard.data_ptr() == bucket.grads_shard.data_ptr():
                    bucket.grads_shard = bucket.grads_shard.clone()

        # Side stream for communication
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        comm_stream.wait_stream(main_stream)

        # Reduce-scatter over distributed process group
        if self.distributed_size > 1:
            with torch.cuda.stream(comm_stream):
                group = self.distributed_process_group
                with _coalescing_manager(group, self.device, async_ops=True) as cm:
                    for bucket in buckets:
                        if self.average_grad_sync and self.nccl_ub:
                            bucket.grads_bucket /= self.distributed_size
                        _coalescing_manager_append_work(
                            cm,
                            reduce_scatter_tensor(
                                bucket.sync_grads_shard,
                                bucket.grads_bucket,
                                op=reduce_op,
                                group=group,
                                async_op=True,
                            ),
                        )
                cm.wait()

        # All-reduce over redundant process group
        if self.redundant_size > 1:
            with torch.cuda.stream(comm_stream):
                group = self.redundant_process_group
                with _coalescing_manager(group, self.device, async_ops=True) as cm:
                    for bucket in buckets:
                        _coalescing_manager_append_work(
                            cm,
                            torch.distributed.all_reduce(
                                bucket.sync_grads_shard,
                                op=reduce_op,
                                group=group,
                                async_op=True,
                            ),
                        )
                cm.wait()

    def _finish_bucket_grad_sync(self) -> None:
        """Wait for any gradient synchronizations that are in progress"""
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        main_stream.wait_stream(comm_stream)
        for bucket_id, bucket in sorted(self._grads_buckets.items()):
            if bucket.status == self.GradientStatus.SYNCING:
                # Accumulate gradient in local shard
                if bucket.grads_shard is None:
                    bucket.grads_shard = bucket.sync_grads_shard
                else:
                    bucket.grads_shard.add_(bucket.sync_grads_shard)
                bucket.grads_bucket = None

                # Reset status
                bucket.status = self.GradientStatus.READY

                # Cached gradient norm has been invalidated
                self._grad_norm = None

    def _try_start_bucket_param_sync(
        self,
        params: Iterable[torch.nn.Parameter] = None,
    ) -> None:
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
            if any(
                bucket.status == self.ParameterStatus.SYNCING
                for bucket in self._params_buckets.values()
            ):
                return
            for bucket_id, bucket in self._params_buckets.items():
                if bucket.status == self.ParameterStatus.SHARDED:
                    params.append(
                        self.parameter(
                            self.state["buckets"][bucket_id].fragments[-1]
                        )
                    )
                    break

        # Find buckets corresponding to params
        bucket_ids = set()
        for param in params:
            bucket_ids.update(
                fragment.bucket_id for fragment in self.state[param]["fragments"]
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

    def _start_bucket_param_sync(self, buckets: List[ParameterBucket]) -> None:
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
            if bucket.params_bucket is not None:
                pass
            elif self.distributed_size == 1:
                bucket.params_bucket = bucket.params_shard
            else:
                shard_size = bucket.params_shard.numel()
                bucket_size = shard_size * self.distributed_size
                bucket.params_bucket = torch.empty(
                    [bucket_size],
                    dtype=bucket.params_shard.dtype,
                    device=bucket.params_shard.device,
                )

        # Side stream for communication
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        comm_stream.wait_stream(main_stream)

        # All-gather over distributed process group
        if self.distributed_size > 1:
            with torch.cuda.stream(comm_stream):
                group = self.distributed_process_group
                with _coalescing_manager(group, self.device, async_ops=True) as cm:
                    for bucket in buckets:
                        _coalescing_manager_append_work(
                            cm,
                            all_gather_into_tensor(
                                bucket.params_bucket,
                                bucket.params_shard,
                                group=group,
                                async_op=True,
                            ),
                        )
                cm.wait()

    def _finish_bucket_param_sync(self) -> None:
        """Wait for any param synchronizations that are in progress"""
        main_stream = torch.cuda.current_stream()
        comm_stream = self._pipeline_streams[-1]
        main_stream.wait_stream(comm_stream)
        for bucket_id, bucket in self._params_buckets.items():
            if bucket.status == self.ParameterStatus.SYNCING:
                bucket.params_shard = None
                bucket.status = self.ParameterStatus.READY

    @contextlib.contextmanager
    def no_sync(
        self,
        greedy_grad_copy: None = False,
    ) -> contextlib.AbstractContextManager:
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

    def grad_sync(self) -> None:
        """Ensure that all gradients are synchronized"""
        for bucket in self.state["buckets"]:
            for fragment in bucket.fragments:
                param = self.parameter(fragment)
                if param.grad is not None:
                    self._grad_copy(param)
                    if not self.contiguous_grad_buffer:
                        self._try_start_bucket_grad_sync(
                            params=[param],
                            ignore_last_bucket=False,
                        )
        self._force_bucket_grad_sync()

    def param_sync(self) -> None:
        """Ensure that all parameters are synchronized"""
        if self.contiguous_param_buffer:
            self._param_copy(self.parameters())
        else:
            while self._params_buckets:
                bucket_id, bucket = next(iter((self._params_buckets.items())))
                for fragment in reversed(self.state["buckets"][bucket_id].fragments):
                    self._param_copy(self.parameter(fragment))
        self._params_buckets.clear()

    @torch.no_grad()
    def _local_grad_norm(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
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
            parameters = list(parameters)
            params_set = set(parameters)
            all_params_set = set()
            for bucket in self.state["buckets"]:
                for fragment in bucket.fragments:
                    all_params_set.add(self.parameter(fragment))
            if not params_set.issubset(all_params_set):
                raise RuntimeError(
                    "Attempted to compute gradient norm for a parameter "
                    "that is not managed by DistributedFusedAdam"
                )
            if params_set == all_params_set:
                parameters = None

        # Group grads by dtype
        grad_groups = collections.defaultdict(list)
        if parameters is None:
            # Compute norm of all local gradients
            for bucket_id, grads_bucket in self._grads_buckets.items():
                state_bucket = self.state["buckets"][bucket_id]
                dtype = state_bucket.grad_sync_dtype
                grad_groups[dtype].append(grads_bucket.grads_shard)
        else:
            # Compute norm of selected local gradients
            for param in parameters:
                if "fragments" not in self.state[param]:
                    continue
                for fragment in self.state[param]["fragments"]:
                    if not fragment.in_local_shard:
                        continue
                    shard_start, shard_end = fragment.shard_range
                    if shard_end <= shard_start:
                        continue
                    bucket_id = fragment.bucket_id
                    grads_bucket = self._grads_buckets[bucket_id]
                    state_bucket = self.state["buckets"][bucket_id]
                    grad_groups[state_bucket.grad_sync_dtype].append(
                        grads_bucket.grads_shard[shard_start:shard_end]
                    )

        # Compute norm of each group of grads
        grad_norm_sq = None
        for grad_group in grad_groups.values():
            grad_group_norm_sq = (
                multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    self._dummy_overflow_buf,
                    [grad_group],
                    False,
                )[0]
                ** 2
            )
            if grad_norm_sq is None:
                grad_norm_sq = grad_group_norm_sq
            else:
                grad_norm_sq += grad_group_norm_sq
        if grad_norm_sq is None:
            grad_norm_sq = torch.zeros([], dtype=torch.float32, device=self.device)

        # Interpret norm as scalar
        grad_norm_sq = grad_norm_sq.to(dtype=torch.float32, device=self.device)
        grad_norm_sq = grad_norm_sq.view([])
        return grad_norm_sq

    def grad_norm(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        norm_type: float = 2.0,
        force: bool = False,
    ) -> torch.Tensor:
        """Gradient norm of parameters in optimizer

        The norm is computed over all gradients together, as if they
        were concatenated into a single vector. All provided
        parameters must be managed by optimizer.

        The computed value is cached to avoid redundant communication.

        Arguments:
            parameters (iterable, optional): an iterable of parameters
                in optimizer (default: all parameters in optimizer).
            norm_type (float, optional): type of the used p-norm
                (default: 2). Only 2-norm is currently supported.
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

    def clip_grad_norm(
        self,
        max_norm: float,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """Clips gradient norm of parameters in optimizer

        The norm is computed over all gradients together, as if they
        were concatenated into a single vector. The scaling is
        deferred until the optimizer step, which should be called
        immediately after this function.

        The computed grad norm is cached to avoid redundant
        communication.

        Arguments:
            max_norm (float): max norm of the gradients
            parameters (iterable, optional): an iterable of parameters
                in optimizer (default: all parameters in optimizer).
            norm_type (float, optional): type of the used
                p-norm (default: 2)

        """
        assert max_norm > 0
        total_norm = self.grad_norm(parameters=parameters, norm_type=norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        self._grad_scale *= clip_coef_clamped
        return total_norm

    def unscale_grads(self, inv_scale: torch.Tensor, *args):
        """Custom unscale function for use by AMP gradient scaler

        Overflow checking is deferred to optimization step.

        Arguments:
            inv_scale (torch.Tensor): factor to multiply gradients

        """
        self._grad_scale *= inv_scale.view([])
        return {self.device: torch.zeros(1, dtype=torch.float32, device=self.device)}

    def step(
        self,
        closure: Optional[Callable] = None,
        *,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
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

        # Make sure params are initialized
        self.init_params()

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
            if grad_scaler_state["stage"] is GradScalerOptState.READY:
                assert grad_scaler._scale is not None
                self._grad_scale /= grad_scaler._scale.view([])
            grad_norm = self.grad_norm()
            found_inf = torch.logical_not(torch.isfinite(grad_norm))
            scaler_state = grad_scaler._per_optimizer_states[id(self)]
            scaler_state["found_inf_per_device"] = {found_inf.device: found_inf.float()}
            if found_inf.item():
                return
        self._grad_scale = self._grad_scale.to(dtype=torch.float32, device=self.device)

        # Initialize buffers for param syncs
        num_buckets = len(self.state["buckets"])
        for bucket_id in reversed(range(num_buckets)):
            self._params_buckets[bucket_id] = self.ParameterBucket()
            params_bucket = self._params_buckets[bucket_id]
            state_bucket = self.state["buckets"][bucket_id]
            shard_size = state_bucket.shard_size
            dtype = state_bucket.dtype
            param_sync_dtype = state_bucket.param_sync_dtype

            if self.contiguous_param_buffer:
                # Construct views into contiguous param buffer
                if not self._param_buffers:
                    self.init_param_buffer()
                bucket_size = state_bucket.bucket_size
                buffer_start = state_bucket.contiguous_buffer_offset
                buffer_end = buffer_start + bucket_size
                param_buffer = self._param_buffers[state_bucket.dtypes()]
                params_bucket.params_bucket = param_buffer[buffer_start:buffer_end]
                bucket_start = self.distributed_rank * shard_size
                bucket_end = bucket_start + shard_size
                params_bucket.params_shard = params_bucket.params_bucket[
                    bucket_start:bucket_end
                ]

            # Initialize param shard buffer
            if self.with_scaled_states:
                # Use FP32 workspace buffer with scaled optimizer state
                params_bucket.params_shard = None
            elif not param_sync_dtype.is_floating_point:
                # Make sure param shard buffer is floating-point
                if (
                    state_bucket.params_shard is not None
                    and dtype.is_floating_point
                ):
                    params_bucket.params_shard = state_bucket.params_shard
                else:
                    params_bucket.params_shard = torch.empty(
                        [shard_size],
                        dtype=self.dtype,
                        device=self.device,
                    )
            else:
                # Allocate param shard buffer if needed
                if params_bucket.params_shard is not None:
                    pass
                elif (
                    state_bucket.params_shard is not None
                    and dtype == param_sync_dtype
                ):
                    params_bucket.params_shard = state_bucket.params_shard
                else:
                    params_bucket.params_shard = torch.empty(
                        [shard_size],
                        dtype=param_sync_dtype,
                        device=self.device,
                    )

        # Apply optimizer step
        self.state["step"] += 1
        overlap_first_bucket = (
            self.distributed_size > 1
            and self.overlap_param_sync
            and self.state["buckets"]
        )
        if overlap_first_bucket:
            # Local step and non-blocking param sync
            # Note: Overlap param sync of first buckets with optimizer
            # step of remaining buckets.

            # Get buckets containing "first" parameter
            first_param = self.parameter(
                self.state["buckets"][-1].fragments[-1]
            )
            first_bucket_ids = sorted(
                fragment.bucket_id
                for fragment in self.state[first_param]["fragments"]
            )

            # Local step and launch param sync for first buckets
            self._local_step(first_bucket_ids)
            self._start_bucket_param_sync(
                self._params_buckets[bucket_id] for bucket_id in first_bucket_ids
            )

            # Local step for remaining buckets
            first_bucket_ids = set(first_bucket_ids)
            self._local_step(
                [
                    bucket_id
                    for bucket_id in range(num_buckets)
                    if bucket_id not in first_bucket_ids
                ]
            )

        else:
            # Local step
            self._local_step(list(range(num_buckets)))

        # Synchronize params
        if self.distributed_size > 1 and self.overlap_param_sync:
            # Asynchronous param sync
            self._try_start_bucket_param_sync()
            for param in self.parameters():
                param._pre_forward_hook_is_enabled = True
        else:
            # Blocking param sync
            self.param_sync()

        return loss

    def _local_step(self, bucket_ids: List[int]) -> None:
        """Apply optimizer step to local shard of parameter buckets

        Arguments:
            bucket_ids (list): bucket indices

        """

        # Implementation with scaled optimizer state
        if self.with_scaled_states:
            self._local_step_with_scaled_states(bucket_ids)
            return

        # Optimized implementation with BF16 params and 16-bit param
        # remainders
        if self.store_param_remainders:
            bf16_rem_buckets = set()
            for bucket_id in bucket_ids:
                state_bucket = self.state["buckets"][bucket_id]
                if state_bucket.param_remainders_shard is not None:
                    bf16_rem_buckets.add(bucket_id)
            if bf16_rem_buckets:
                self._local_step_with_param_remainders(sorted(bf16_rem_buckets))
            bucket_ids = [
                bucket_id
                for bucket_id in bucket_ids
                if bucket_id not in bf16_rem_buckets
            ]
            if not bucket_ids:
                return

        # Find param fragments for each bucket
        buffers = collections.defaultdict(list)  # p_in, m, v, g, p_out
        for bucket_id in bucket_ids:
            state_bucket = self.state["buckets"][bucket_id]
            grads_bucket = self._grads_buckets[bucket_id]
            params_bucket = self._params_buckets[bucket_id]

            # Optimizer state buffers for local shard
            fragments = state_bucket.fragments
            exp_avg = state_bucket.exp_avg_shard
            exp_avg_sq = state_bucket.exp_avg_sq_shard
            grads = grads_bucket.grads_shard
            params_out = params_bucket.params_shard

            # Find param fragments in local shard
            for fragment in fragments:
                if not fragment.in_local_shard:
                    continue
                shard_start, shard_end = fragment.shard_range
                if shard_end <= shard_start:
                    continue
                shard_range = slice(shard_start, shard_end)
                if state_bucket.params_shard is None:
                    param = self.parameter(fragment)
                    param_range = slice(*fragment.shard_param_range)
                    param_fragment = param.detach().view(-1)[param_range]
                    param_fragment = param_fragment.to(
                        dtype=state_bucket.dtype, device=self.device
                    )
                else:
                    params_shard = state_bucket.params_shard
                    param_fragment = params_shard[shard_range]
                buffers_key = (
                    fragment.param_group_id,
                    state_bucket.dtype,
                    state_bucket.grad_sync_dtype,
                    state_bucket.param_sync_dtype,
                )
                buffers[buffers_key].append(
                    [
                        param_fragment,
                        exp_avg[shard_range],
                        exp_avg_sq[shard_range],
                        grads[shard_range],
                        params_out[shard_range],
                    ]
                )

        # Apply optimizer step to each param group
        for (group_id, _, _, _), group_buffers in buffers.items():
            group = self.param_groups[group_id]
            beta1, beta2 = group["betas"]
            multi_tensor_applier(
                distributed_adam_cuda.multi_tensor_fused_adam,
                self._dummy_overflow_buf,
                list(zip(*group_buffers)),
                self._grad_scale,
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                self.state["step"],
                1 if self.adam_w_mode else 0,
                1 if group["bias_correction"] else 0,
                group["weight_decay"],
            )

        # Make sure param sync buffer has correct dtype
        self._check_params_shard_dtypes(
            {
                bucket_id: self._params_buckets[bucket_id]
                for bucket_id in bucket_ids
            }
        )

    def _local_step_with_param_remainders(
        self,
        bucket_ids: List[int],
    ) -> None:
        """Apply optimizer step to local shard of parameter bucket

        This is an experimental implementation that expects
        store_params=False and store_param_remainders=True. The
        optimizer dtype must be FP32 and the params must all be BF16
        and GPU.

        Arguments:
            bucket_ids (list): bucket indices

        """

        # Find param fragments for each bucket
        buffers = collections.defaultdict(list)  # p_in, p_rem, m, v, g, p_out
        for bucket_id in bucket_ids:
            state_bucket = self.state["buckets"][bucket_id]
            grads_bucket = self._grads_buckets[bucket_id]
            params_bucket = self._params_buckets[bucket_id]

            # State buffers for local shard
            fragments = state_bucket.fragments
            param_remainders_shard = state_bucket.param_remainders_shard
            exp_avg = state_bucket.exp_avg_shard
            exp_avg_sq = state_bucket.exp_avg_sq_shard
            grads = grads_bucket.grads_shard
            params_out = params_bucket.params_shard

            # Find param fragments in local shard
            for fragment in fragments:
                if not fragment.in_local_shard:
                    continue
                shard_start, shard_end = fragment.shard_range
                if shard_end <= shard_start:
                    continue
                shard_range = slice(shard_start, shard_end)
                buffers_key = (
                    fragment.param_group_id,
                    state_bucket.grad_sync_dtype,
                )
                param = self.parameter(fragment)
                param_range = slice(*fragment.shard_param_range)
                param_fragment = param.detach().view(-1)[param_range]
                param_fragment = param_fragment.to(
                    dtype=torch.bfloat16, device=self.device
                )
                buffers[buffers_key].append(
                    [
                        param_fragment,
                        param_remainders_shard[shard_range],
                        exp_avg[shard_range],
                        exp_avg_sq[shard_range],
                        grads[shard_range],
                        params_out[shard_range],
                    ]
                )

        # Apply optimizer step to each param group
        for (group_id, _), group_buffers in buffers.items():
            group = self.param_groups[group_id]
            beta1, beta2 = group["betas"]
            multi_tensor_applier(
                distributed_adam_cuda.multi_tensor_fused_adam_with_param_remainders,
                self._dummy_overflow_buf,
                list(zip(*group_buffers)),
                self._grad_scale,
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                self.state["step"],
                1 if self.adam_w_mode else 0,
                1 if group["bias_correction"] else 0,
                group["weight_decay"],
            )

        # Make sure param sync buffer has correct dtype
        self._check_params_shard_dtypes(
            {
                bucket_id: self._params_buckets[bucket_id]
                for bucket_id in bucket_ids
            }
        )

    @torch.no_grad()
    def _local_step_with_scaled_states(
        self,
        bucket_ids: List[int],
    ) -> None:
        for bucket_id in bucket_ids:
            state_bucket = self.state["buckets"][bucket_id]
            grads_bucket = self._grads_buckets[bucket_id]
            params_bucket = self._params_buckets[bucket_id]
            params_bucket.params_shard = torch.empty_like(
                state_bucket.params_shard,
                dtype=torch.float32,
            )

            # Find param fragments in local shard
            group_buffers = collections.defaultdict(list)  # p_in, m, v, g, p_out
            scaled_buffers = []
            unscaled_buffers = []
            buffer_scales = []
            for fragment in state_bucket.fragments:
                if not fragment.in_local_shard:
                    continue
                shard_start, shard_end = fragment.shard_range
                if shard_end <= shard_start:
                    continue
                shard_range = slice(shard_start, shard_end)
                param_group_id = fragment.param_group_id
                param_id = fragment.param_id
                scaled_param = state_bucket.params_shard[shard_range]
                scaled_exp_avg = state_bucket.exp_avg_shard[shard_range]
                scaled_exp_avg_sq = state_bucket.exp_avg_sq_shard[shard_range]
                grads = grads_bucket.grads_shard[shard_range]
                param = params_bucket.params_shard[shard_range]
                exp_avg = torch.empty_like(scaled_exp_avg, dtype=torch.float32)
                exp_avg_sq = torch.empty_like(scaled_exp_avg_sq, dtype=torch.float32)
                scales = self._state_scales[(param_group_id, param_id, bucket_id)]
                group_buffers[param_group_id].append(
                    (param, exp_avg, exp_avg_sq, grads, param)
                )
                scaled_buffers.extend(
                    (scaled_param, scaled_exp_avg, scaled_exp_avg_sq)
                )
                unscaled_buffers.extend((param, exp_avg, exp_avg_sq))
                buffer_scales.extend(
                    (scales["param"], scales["exp_avg"], scales["exp_avg_sq"])
                )

            # Unscale optimizer state
            _multi_tensor_copy(
                scaled_buffers,
                unscaled_buffers,
                dummy_overflow_buf=self._dummy_overflow_buf,
            )
            for buf, scale in zip(unscaled_buffers, buffer_scales):
                buf.mul_(scale)

            # Apply optimizer step to each param group
            for group_id, buffers in group_buffers.items():
                group = self.param_groups[group_id]
                beta1, beta2 = group["betas"]
                multi_tensor_applier(
                    distributed_adam_cuda.multi_tensor_fused_adam,
                    self._dummy_overflow_buf,
                    list(zip(*buffers)),
                    self._grad_scale,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    self.state["step"],
                    1 if self.adam_w_mode else 0,
                    1 if group["bias_correction"] else 0,
                    group["weight_decay"],
                )
            del group_buffers

            # Make sure param sync buffer has correct dtype
            self._check_params_shard_dtypes({bucket_id: params_bucket})

            # Scale optimizer state
            for buf, scale in zip(unscaled_buffers, buffer_scales):
                self._apply_state_scale(buf, scale)
            _multi_tensor_copy(
                unscaled_buffers,
                scaled_buffers,
                dummy_overflow_buf=self._dummy_overflow_buf,
            )
            del scaled_buffers, unscaled_buffers, buffer_scales

    @torch.no_grad()
    def _check_params_shard_dtypes(
        self,
        params_buckets: Dict[int, ParameterBucket],
    ) -> None:
        """Make sure local shards of parameters are in expected datatypes

        The Adam kernel only supports floating-point datatypes. If we
        want to perform parameter synchronization with
        non-floating-point dtypes, we need to allocate temporary
        buffers that can accommodate the Adam kernel. This function is
        responsible for converting these temporary buffers to the
        parameter synchronization datatype.

        """

        # Find param shards that require dtype conversion
        buffers_in = []
        buffers_out = []
        for bucket_id, param_bucket in params_buckets.items():

            # Check if param shard is already in expected dtype
            state_bucket = self.state["buckets"][bucket_id]
            param_sync_dtype = state_bucket.param_sync_dtype
            if param_bucket.params_shard.dtype == param_sync_dtype:
                continue

            # Allocate buffer with required dtype
            buffer_in = param_bucket.params_shard
            buffer_out = torch.empty_like(
                param_bucket.params_shard,
                dtype=param_sync_dtype,
            )
            param_bucket.params_shard = buffer_out

            if (
                torch.is_floating_point(buffer_in)
                and torch.is_floating_point(buffer_out)
            ):
                # Cast between floating-point dtypes
                buffers_in.append(buffer_in)
                buffers_out.append(buffer_out)
            else:
                # Copy most significant bytes for non-floating-point
                # dtypes
                # Note: Assume dtypes are little-endian
                in_bytes = buffer_in.unsqueeze(-1).view(torch.uint8)
                out_bytes = buffer_out.unsqueeze(-1).view(torch.uint8)
                copy_size = min(in_bytes.size(-1), out_bytes.size(-1))
                buffers_in.append(in_bytes[..., -copy_size:])
                buffers_out.append(out_bytes[..., -copy_size:])
                if copy_size < out_bytes.size(-1):
                    out_bytes[..., :-copy_size].zero_()

        # Perform dtype conversions
        _multi_tensor_copy(
            buffers_in,
            buffers_out,
            dummy_overflow_buf=self._dummy_overflow_buf,
        )

    @torch.no_grad()
    def _apply_state_scale(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Compute and apply scaling factor for scaled optimizer state

        The scaling factor is chosen to maximize the dynamic range
        while avoiding numerical overflows. The returned tensors are
        the scale (used to unscale the optimizer state) and the
        scale-reciprocal (used to generate the scaled optimizer
        state). The input tensors are updated in-place.

        """
        if not hasattr(self, "_max_scaled_state"):
            self._max_scaled_state = torch.full(
                [1],
                torch.finfo(self.dtype).max / 2,
                dtype=torch.float32,
                device=self.device,
            )
        min_val, max_val = torch.aminmax(tensor)
        absmax = torch.maximum(-min_val, max_val)
        absmax = absmax.to(dtype=torch.float32, device=self.device)
        torch.div(absmax, self._max_scaled_state, out=scale)
        rscale = torch.where(scale > 0, scale.reciprocal(), 0.0)
        tensor.mul_(rscale)

    def state_dict(
        self,
        *,
        state_dict_format: Optional[int] = None,
        gather_on_root: Optional[bool] = None,
    ) -> Optional[dict]:
        """Get dictionary containing optimizer state

        All ranks in the process group must call this function since
        it performs communication. The same optimizer state is
        returned on all ranks.

        Arguments:
            state_dict_format (int, optional): Tag for custom or
                deprecated state dict format.
            gather_on_root (bool, optional): Option for deprecated v1
                format.

        """

        # Default state dict format
        if state_dict_format is None:
            state_dict_format = 2

        # Construct state dict
        state_dict = None
        if state_dict_format == 1:
            # Deprecated v1 format
            kwargs = {}
            if gather_on_root is not None:
                kwargs["gather_on_root"] = gather_on_root
            state_dict = self._state_dict_v1(**kwargs)
        elif state_dict_format == 2:
            # Default v2 format
            state_dict = self._state_dict_v2()
        else:
            # Unrecognized format
            raise ValueError(f"Unrecognized state dict format ({state_dict_format})")

        # Add format tag to state dict
        if state_dict is not None:
            state_dict["format"] = state_dict_format

        return state_dict

    def _state_dict_v1(self, gather_on_root: bool = True) -> Optional[dict]:
        """Get dictionary containing optimizer state (deprecated v1 format)

        Default behavior is to perform communication so that the
        entire optimizer state is returned on the root rank in the
        process group. In this case, all ranks in the process group
        must enter this function and no value is returned on non-root
        ranks.

        Arguments:
            gather_on_root (bool, optional): Gather state from all
                ranks on the root rank (default: True)

        """
        warnings.warn(
            "Making optimizer state dictionary in deprecated v1 format. "
            "Future support is not guaranteed."
        )
        if self.with_scaled_states:
            raise NotImplementedError(
                "Deprecated v1 format does not support scaled state"
            )

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
        chunk_size = (
            self.default_shard_size * torch.finfo(self.grad_sync_dtype).bits // 8
        )
        if self.distributed_rank == 0:
            gathered_state_bytes = [
                torch.empty([size], dtype=torch.uint8, device="cpu")
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
                        gathered_chunks_buffers[stream_id][
                            i * chunk_size : (i + 1) * chunk_size
                        ]
                        for i in range(self.distributed_size)
                    ]
                else:
                    chunk = chunk_buffers[stream_id]

                # Copy to GPU
                if self.distributed_rank != 0 and offset < local_state_size:
                    local_chunk_size = min(chunk_size, local_state_size - offset)
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
                            no_copy_kwarg = {"no_copy": True}
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
                            dst = gathered_state_bytes[rank][
                                rank_chunk_start:rank_chunk_end
                            ]
                            dst.copy_(src, non_blocking=True)

        # Synchronize GPU
        for stream in self._pipeline_streams:
            main_stream.wait_stream(stream)
        main_stream.synchronize()

        # Return gathered state data on root rank
        if self.distributed_rank == 0:
            return {"gathered_states": gathered_state_bytes}
        else:
            return None

    @torch.no_grad()
    def _state_dict_v2(self) -> Optional[dict]:
        """Get dictionary containing optimizer state (default v2 format)

        All ranks in the process group must call this function since
        it performs communication. The same optimizer state is
        returned on all ranks.

        """

        # Make sure params are initialized
        self.init_params()

        # Finish any asynchronous communication
        self.grad_sync()
        self.param_sync()

        # Output tensor format
        dtype = torch.float32 if self.with_scaled_states else self.dtype
        device = torch.device("cpu")

        # Get state dict from base class
        state_dict = super().state_dict()
        state_dict["state"] = {"step": state_dict["state"]["step"]}

        # Initialize state dict with CPU buffers
        for param in self.parameters():
            # Get param index in state dict
            fragment = self.state[param]["fragments"][0]
            param_group_id = fragment.param_group_id
            param_id = fragment.param_id
            index = state_dict["param_groups"][param_group_id]["params"][param_id]

            # Construct CPU buffers with optimizer state
            state_dict["state"][index] = dict(
                param=torch.zeros_like(param, dtype=dtype, device=device),
                exp_avg=torch.zeros_like(param, dtype=dtype, device=device),
                exp_avg_sq=torch.zeros_like(param, dtype=dtype, device=device),
            )

        # Workspace buffers for gathering shards on root rank
        num_buckets = len(self.state["buckets"])
        max_bucket_size = max(bucket.bucket_size for bucket in self.state["buckets"])
        bucket_buffers = [
            torch.empty(
                [max_bucket_size],
                dtype=dtype,
                device=self.device,
            )
            for _ in range(self.pipeline_size)
        ]
        if self.store_param_remainders:
            max_shard_size = max(bucket.shard_size for bucket in self.state["buckets"])
            shard_bf16_buffers = [
                torch.empty([max_shard_size], dtype=torch.bfloat16, device=self.device)
                for _ in range(self.pipeline_size)
            ]

        # Synchronize streams
        main_stream = torch.cuda.current_stream()
        for stream in self._pipeline_streams:
            stream.wait_stream(main_stream)

        def get_workspace_shard(bucket_id: int) -> torch.Tensor:
            """Workspace buffer for local shard"""
            bucket = self.state["buckets"][bucket_id]
            shard_size = bucket.shard_size
            stream_id = bucket_id % self.pipeline_size
            shard_range = slice(
                shard_size * self.distributed_rank,
                shard_size * (self.distributed_rank + 1),
            )
            return bucket_buffers[stream_id][shard_range]

        def unscale_shard(
            bucket_id: int,
            shard: torch.Tensor,
            state_key: str,
        ) -> torch.Tensor:
            """Unscale local shard if needed

            If state buffers are scaled, then the shard is unscaled
            and output to a workspace buffer. Otherwise, the shard is
            immediately returned.

            """
            if not self.with_scaled_states:
                return shard
            out = get_workspace_shard(bucket_id)
            bucket = self.state["buckets"][bucket_id]
            stream_id = bucket_id % self.pipeline_size
            stream = self._pipeline_streams[stream_id]
            with torch.cuda.stream(stream):
                for fragment in bucket.fragments:
                    if not fragment.in_local_shard:
                        continue
                    param_group_id = fragment.param_group_id
                    param_id = fragment.param_id
                    shard_range = slice(*fragment.shard_range)
                    scale = self._state_scales[(param_group_id, param_id, bucket_id)][state_key]
                    out[shard_range].copy_(shard[shard_range]).mul_(scale)
            return out

        def pack_param_shard(bucket_id: int) -> torch.Tensor:
            """Pack local shard of param values into contiguous buffer"""

            # Stream objects
            stream_id = bucket_id % self.pipeline_size
            stream = self._pipeline_streams[stream_id]

            # Bucket objects
            bucket = self.state["buckets"][bucket_id]
            shard_size = bucket.shard_size

            # Case 1: Param state is already packed
            if bucket.params_shard is not None:
                return unscale_shard(bucket_id, bucket.params_shard, "param")

            # Case 2: Pack BF16 model params with 16-bit remainders
            if bucket.param_remainders_shard is not None:
                with torch.cuda.stream(stream):
                    # Pack bf16 param values
                    shard_bf16 = shard_bf16_buffers[stream_id][:shard_size]
                    buffers_in = []
                    buffers_out = []
                    for fragment in bucket.fragments:
                        if not fragment.in_local_shard:
                            continue
                        param_range = slice(*fragment.shard_param_range)
                        shard_range = slice(*fragment.shard_range)
                        param = self.parameter(fragment)
                        buffers_in.append(param.view(-1)[param_range])
                        buffers_out.append(shard_bf16[shard_range])
                    _multi_tensor_copy(
                        buffers_in,
                        buffers_out,
                        dummy_overflow_buf=self._dummy_overflow_buf,
                    )

                    # Reconstruct fp32 from bf16 and remainders
                    shard_fp32 = get_workspace_shard(bucket_id)
                    _bf16_rem_to_fp32(
                        shard_bf16,
                        bucket.param_remainders_shard,
                        shard_fp32,
                    )
                    return shard_fp32

            # Case 3: Pack model params
            with torch.cuda.stream(stream):
                shard = get_workspace_shard(bucket_id)
                buffers_in = []
                buffers_out = []
                for fragment in bucket.fragments:
                    if not fragment.in_local_shard:
                        continue
                    param_range = slice(*fragment.shard_param_range)
                    shard_range = slice(*fragment.shard_range)
                    param = self.parameter(fragment)
                    buffers_in.append(param.view(-1)[param_range])
                    buffers_out.append(shard[shard_range])
                _multi_tensor_copy(
                    buffers_in,
                    buffers_out,
                    dummy_overflow_buf=self._dummy_overflow_buf,
                )
                return shard

        def start_all_gather(bucket_id: int, shard: torch.Tensor) -> None:
            """Launch all-gather on bucket shards

            Communication is done on main stream to ensure consistent
            ordering.

            """

            # Stream objects
            stream_id = bucket_id % self.pipeline_size
            stream = self._pipeline_streams[stream_id]

            # Workspace buffer
            bucket = self.state["buckets"][bucket_id]
            bucket_size = bucket.bucket_size
            bucket_buffer = bucket_buffers[stream_id][:bucket_size]

            # All-gather shards
            main_stream.wait_stream(stream)
            all_gather_into_tensor(
                bucket_buffer,
                shard,
                group=self.distributed_process_group,
            )
            stream.wait_stream(main_stream)

        def finish_all_gather(bucket_id: int, state_dict_key: str) -> None:
            """Finish all-gather on bucket shards

            Data is copied into state dict CPU buffers.

            Splitting the NCCL all-gather and the CPU memcpys into
            separate stages helps achieve good overlap when kernel
            launches are serialized with
            CUDA_DEVICE_MAX_CONNECTIONS=1. In particular, the pipeline
            calls start_all_gather(bucket_id+1) before
            finish_all_gather(bucket_id).

            """

            # Stream objects
            stream_id = bucket_id % self.pipeline_size
            stream = self._pipeline_streams[stream_id]

            # Bucket objects
            bucket = self.state["buckets"][bucket_id]
            bucket_size = bucket.bucket_size
            bucket_buffer = bucket_buffers[stream_id][:bucket_size]

            # Update state dict
            with torch.cuda.stream(stream):
                for fragment in bucket.fragments:
                    param_range = slice(*fragment.param_range)
                    bucket_range = slice(*fragment.bucket_range)
                    param_group_id = fragment.param_group_id
                    param_id = fragment.param_id
                    index = state_dict["param_groups"][param_group_id]["params"][
                        param_id
                    ]
                    state_buffer = state_dict["state"][index][state_dict_key]
                    state_fragment = state_buffer.view(-1)[param_range]
                    bucket_fragment = bucket_buffer[bucket_range]
                    state_fragment.copy_(bucket_fragment, non_blocking=True)

        # All-gather param state
        for bucket_id in range(num_buckets):
            shard = pack_param_shard(bucket_id)
            start_all_gather(bucket_id, shard)
            if bucket_id > 0:
                finish_all_gather(bucket_id - 1, "param")
            if bucket_id == num_buckets - 1:
                finish_all_gather(bucket_id, "param")

        # All-gather exp_avg state
        for bucket_id in range(num_buckets):
            shard = unscale_shard(
                bucket_id,
                self.state["buckets"][bucket_id].exp_avg_shard,
                "exp_avg",
            )
            start_all_gather(bucket_id, shard)
            if bucket_id > 0:
                finish_all_gather(bucket_id - 1, "exp_avg")
            if bucket_id == num_buckets - 1:
                finish_all_gather(bucket_id, "exp_avg")

        # All-gather exp_avg_sq state
        for bucket_id in range(num_buckets):
            shard = unscale_shard(
                bucket_id,
                self.state["buckets"][bucket_id].exp_avg_sq_shard,
                "exp_avg_sq",
            )
            start_all_gather(bucket_id, shard)
            if bucket_id > 0:
                finish_all_gather(bucket_id - 1, "exp_avg_sq")
            if bucket_id == num_buckets - 1:
                finish_all_gather(bucket_id, "exp_avg_sq")

        # Synchronize GPU and return
        for stream in self._pipeline_streams:
            main_stream.wait_stream(stream)
        main_stream.synchronize()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state"""

        # Figure out state dict format
        state_dict_format = state_dict.pop("format", None)
        if state_dict_format is None:
            if "buckets" in state_dict or "gathered_states" in state_dict:
                state_dict_format = 1
            else:
                state_dict_format = 2

        # Load state dict
        if state_dict_format == 1:
            # Deprecated v1 format
            self._load_state_dict_v1(state_dict)
        elif state_dict_format == 2:
            # Default v2 format
            self._load_state_dict_v2(state_dict)
        else:
            # Unrecognized format
            raise ValueError(f"Unrecognized state dict format ({state_dict_format})")

    def _load_state_dict_v1(self, state_dict: dict) -> None:
        """Load optimizer state (deprecated v1 format)

        Parallel configuration (e.g. process group sizes) and
        optimizer options must match between saving and loading the
        optimizer state.

        """
        warnings.warn(
            "Loading checkpoint in deprecated v1 format. "
            "Future support is not guaranteed."
        )
        if self.with_scaled_states:
            raise NotImplementedError(
                "Deprecated v1 format does not support scaled state"
            )

        # Get state dict for current rank
        if "gathered_states" in state_dict:
            # Deallocate distributed optimizer state to reduce GPU
            # memory usage
            if "buckets" in self.state:
                del self.state["buckets"]

            # Get state for current rank and parse byte string
            state_bytes = state_dict["gathered_states"][self.distributed_rank]
            state_bytes = io.BytesIO(state_bytes.numpy())
            state_dict = torch.load(state_bytes)

        # Load state dict
        super().load_state_dict(state_dict)

        # Handle old state dicts without per-bucket dtypes
        for bucket in self.state["buckets"]:
            if getattr(bucket, "dtype", None) is None:
                bucket.dtype = self.dtype
            if getattr(bucket, "grad_sync_dtype", None) is None:
                bucket.grad_sync_dtype = self.grad_sync_dtype
            if getattr(bucket, "param_sync_dtype", None) is None:
                bucket.param_sync_dtype = self.param_sync_dtype

            if bucket.params_shard is not None:
                bucket.params_shard = bucket.params_shard.to(self.device)
            if bucket.param_remainders_shard is not None:
                bucket.param_remainders_shard = bucket.param_remainders_shard.to(self.device)
            bucket.exp_avg_shard = bucket.exp_avg_shard.to(self.device)
            bucket.exp_avg_sq_shard = bucket.exp_avg_sq_shard.to(self.device)

    @torch.no_grad()
    def _load_state_dict_v2(self, state_dict: dict) -> None:
        """Load optimizer state (default v2 format)

        The parallel configuration and optimizer options are allowed
        to differ between saving and loading the model.

        """

        # Make sure params are initialized
        self.init_params()

        # Finish any asynchronous communication
        self.grad_sync()
        self.param_sync()

        # Load general state
        # Note: State includes bucketing scheme (e.g.
        # self.state["buckets"] and self.state[param]["fragments"]).
        # This was needed for v1 checkpoints, but not for v2. As a
        # kludge, we temporarily set state to dummy dict to avoid
        # messing up the bucketing scheme.
        state = self.state
        self.state = {}
        super().load_state_dict(
            {
                "state": {},
                "param_groups": state_dict["param_groups"],
            }
        )
        self.state = state
        self.state["step"] = state_dict["state"]["step"]

        # Load state for each param
        for param in self.parameters():
            # Get param index in state dict
            fragment = self.state[param]["fragments"][0]
            param_id = fragment.param_id
            param_group_id = fragment.param_group_id
            index = state_dict["param_groups"][param_group_id]["params"][param_id]

            # Buffers in state dict
            param_state = state_dict["state"][index]["param"].view(-1)
            exp_avg = state_dict["state"][index]["exp_avg"].view(-1)
            exp_avg_sq = state_dict["state"][index]["exp_avg_sq"].view(-1)

            # Copy to local shard of state buckets
            for fragment in self.state[param]["fragments"]:
                if not fragment.in_local_shard:
                    continue
                bucket_id = fragment.bucket_id
                bucket = self.state["buckets"][bucket_id]
                param_range = slice(*fragment.shard_param_range)
                shard_range = slice(*fragment.shard_range)
                if self.with_scaled_states:
                    scales = self._state_scales[(param_group_id, param_id, bucket_id)]
                    temp = torch.empty_like(
                        param_state[param_range],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    temp.copy_(param_state[param_range], non_blocking=True)
                    self._apply_state_scale(temp, scales["param"])
                    bucket.params_shard[shard_range].copy_(temp)
                    temp.copy_(exp_avg[param_range], non_blocking=True)
                    self._apply_state_scale(temp, scales["exp_avg"])
                    bucket.exp_avg_shard[shard_range].copy_(temp)
                    temp.copy_(exp_avg_sq[param_range], non_blocking=True)
                    self._apply_state_scale(temp, scales["exp_avg_sq"])
                    bucket.exp_avg_sq_shard[shard_range].copy_(temp)
                else:
                    if bucket.params_shard is not None:
                        bucket.params_shard[shard_range].copy_(
                            param_state[param_range],
                            non_blocking=True,
                        )
                    if bucket.param_remainders_shard is not None:
                        param_state_int16 = param_state.unsqueeze(-1).view(torch.int16)
                        bucket.param_remainders_shard[shard_range].copy_(
                            param_state_int16[param_range, 0],
                            non_blocking=True,
                        )
                    bucket.exp_avg_shard[shard_range].copy_(
                        exp_avg[param_range],
                        non_blocking=True,
                    )
                    bucket.exp_avg_sq_shard[shard_range].copy_(
                        exp_avg_sq[param_range],
                        non_blocking=True,
                    )

        # Synchronize GPU
        torch.cuda.current_stream().synchronize()
