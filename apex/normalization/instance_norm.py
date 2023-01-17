from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.modules.batchnorm import _NormBase

from nvfuser._C import DataType, Fusion, FusionDefinition, Scalar, Tensor


NamedAxis = Enum("NamedAxis", ["BATCH", "CHANNEL"])


def torch2datatype(dt: torch.dtype) -> Optional[DataType]:
    """Translate between PyTorch and NVFuser element types.

    Returns `None` if the type cannot be translated.
    """
    return {
        bool: DataType.Bool,
        torch.float16: DataType.Half,
        torch.bfloat16: DataType.BFloat16,
        torch.float32: DataType.Float,
        torch.float64: DataType.Double,
        torch.int32: DataType.Int32,
        torch.int64: DataType.Int,
        torch.bool: DataType.Bool,
        torch.complex64: DataType.ComplexFloat,
        torch.complex128: DataType.ComplexDouble,
    }.get(dt)


def norm_fusion_forward(
    fd: FusionDefinition,
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    eps: Scalar,
    use_input_stats: bool,
    momentum: Scalar,
    channels_last: bool,
    x_datatype: DataType,
    extent: torch.Size,
    unbiased: bool = False,
    *,
    stat_axes: List[NamedAxis],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Modify FusionDefinition to add a generic normalization layer (forward).

    This can be used to construct a BatchNorm, GroupNorm, InstanceNorm, or
    LayerNorm network by indicating different sets of axes to preserve.

    BatchNorm: `stat_axes = [NamedAxis.CHANNEL]`
    LayerNorm: `stat_axes = [NamedAxis.BATCH]`
    InstanceNorm: `stat_axes = [NamedAxis.BATCH, NamedAxis.CHANNEL]`

    Args:
        fd: An initialized FusionDefinition.
        x: An input NVFuser tensor.
        weight: If given, multiply normed output by this `Tensor`. It should be
            one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
            zero-dimensional otherwise. It will be broadcast along all other
            dimensions.
        bias: If given, add this `Tensor` to normed output. It should be
            one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
            zero-dimensional otherwise. It will be broadcast along all other
            dimensions.
        running_mean: If given, a running mean estimate that will be modified
            in place.
        running_var: If given, a running variance estimate that will be
            modified in place.
        eps: Amount to regularize the square root needed to convert variance to
            standard deviation.
        use_input_stats: Whether to compute the stats of this batch or to
            _only_ use the provided running_mean and running_var.
        momentum: Momentum for exponentially weighted moving average of running
            stats.
        channels_last: Whether channels are in position -1 (`True`) or 1
            (`False`).
        x_datatype: :class:'DataType' of input :class:'Tensor' `x`
        extent: Size of the input.
        unbiased: Whether to use unbiased variance for computing current batch
            statistics. Note that unbiased estimates are always used for
            running variance updates, regardless of this argument's value.
        stat_axes: A list of `NamedAxis` objects indicating a combination of
            axes with which to index the computed statistics. This can be used
            to implement multiple types of normalization layers, since most of
            those differ only in which axes are reduced over.
    Returns:
        The normalized output, as well as mean and 1/std. Note that
        `fd.add_output` is _not_ called by this function.
    """
    assert not (
        (running_var is None) ^ (running_mean is None)
    ), "Iff running mean or var is given, the other should be"

    batch_dim = 0
    num_dims = len(extent)
    channel_dim = num_dims - 1 if channels_last else 1

    stat_dims = []
    # Running stats will be kept possibly for channel but never by instance, so
    # we will reduce along batch_dim before updating running stats.
    stat_dims_nobatch = []
    num_stats = 1
    if NamedAxis.BATCH in stat_axes:
        stat_dims.append(batch_dim)
        num_stats *= extent[batch_dim]
    if NamedAxis.CHANNEL in stat_axes:
        stat_dims.append(channel_dim)
        stat_dims_nobatch.append(channel_dim)
        num_stats *= extent[channel_dim]
    x_reduction_axes = [ax for ax in range(num_dims) if ax not in stat_dims]
    num_features = extent.numel() // num_stats

    batch_size = fd.define_constant(extent[batch_dim])

    if use_input_stats or running_mean is None:
        # In NVFuser Python we pass correction=1 to request unbiased variance calculation
        x_var, x_mean = fd.ops.var_mean(x, x_reduction_axes, int(unbiased))
        if running_mean is not None:
            one = fd.define_constant(1.0)
            rev_momentum = fd.ops.sub(one, momentum)

            # do running mean with momentum
            current_mean_hat = fd.ops.mul(x_mean, momentum)
            mean_hat = fd.ops.mul(running_mean, rev_momentum)
            new_mean_hat = fd.ops.add(mean_hat, current_mean_hat)

            # If computing stats for each instance, we don't want to keep those
            # for our running mean calculation, so we sum them here
            new_mean_sum = (
                fd.ops.sum(new_mean_hat, [0])
                if NamedAxis.BATCH in stat_axes
                else new_mean_hat
            )

            rev_batch_size = fd.ops.reciprocal(batch_size)
            new_mean_channels_only = fd.ops.mul(new_mean_sum, rev_batch_size)
            if x_datatype in [DataType.Half, DataType.BFloat16]:
                new_mean_channels_only = fd.ops.cast(new_mean_channels_only, x_datatype)
            fd.add_output(new_mean_channels_only, alias_input=running_mean)

            # running var calculation
            x_var_unbiased = x_var
            if not unbiased:
                # multiply by correction to go from biased to unbiased estimate
                b2ub = fd.define_constant(num_features / (num_features - 1))
                x_var_unbiased = fd.ops.mul(x_var, b2ub)

            current_var_hat = fd.ops.mul(x_var_unbiased, momentum)
            var_hat = fd.ops.mul(running_var, rev_momentum)
            new_var_hat = fd.ops.add(var_hat, current_var_hat)

            # See above about reducing over batch dim for running stats
            new_var_sum = (
                fd.ops.sum(new_var_hat, [0])
                if NamedAxis.BATCH in stat_axes
                else new_var_hat
            )

            new_var_channels_only = fd.ops.mul(new_var_sum, rev_batch_size)
            if x_datatype in [DataType.Half, DataType.BFloat16]:
                new_var_channels_only = fd.ops.cast(new_var_channels_only, x_datatype)
            fd.add_output(new_var_channels_only, alias_input=running_var)

        mean = x_mean
        mean_bcast = fd.ops.broadcast_in_dim(mean, extent, stat_dims)
        x_sub_mean = fd.ops.sub(x, mean_bcast)

        var_eps = fd.ops.add(x_var, eps)
        invstd = fd.ops.rsqrt(var_eps)
        invstd_bcast = fd.ops.broadcast_in_dim(invstd, extent, stat_dims)

        x_normed = fd.ops.mul(x_sub_mean, invstd_bcast)

    else:  # This is inference mode with running stats
        assert running_mean is not None
        r_mean_bcast = fd.ops.broadcast_in_dim(running_mean, extent, stat_dims_nobatch)
        x_sub_mean = fd.ops.sub(x, r_mean_bcast)

        var_eps = fd.ops.add(running_var, eps)
        invstd = fd.ops.rsqrt(var_eps)
        invstd_bcast = fd.ops.broadcast_in_dim(invstd, extent, stat_dims_nobatch)

        mean = running_mean
        x_normed = fd.ops.mul(x_sub_mean, invstd_bcast)

    if weight is not None:
        weight_bcast = fd.ops.broadcast_in_dim(weight, extent, stat_dims_nobatch)
        x_normed = fd.ops.mul(x_normed, weight_bcast)
    if bias is not None:
        bias_bcast = fd.ops.broadcast_in_dim(bias, extent, stat_dims_nobatch)
        x_normed = fd.ops.add(x_normed, bias_bcast)

    return x_normed, mean, invstd


def norm_fusion_backward(
    fd: FusionDefinition,
    x: Tensor,
    grad_output: Tensor,
    mean: Optional[torch.Tensor],
    invstd: torch.Tensor,
    inputs: List[torch.Tensor],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    use_input_stats: bool,
    channels_last: bool,
    x_datatype: DataType,
    extent: torch.Size,
    *,
    stat_axes: List[NamedAxis],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Modify FusionDefinition to add a generic normalization layer (backward).

    Args:
        fd: An initialized FusionDefinition.
        x: An input NVFuser tensor.
        grad_output: NVFuser tensor representing gradient of loss with respect
            to downstream activation (typical input to backward()).
        mean: The mean used in the forward normalization.
        invstd: The reciprocal of standard deviation used in the forward normalization.
        inputs: A list of :class:'torch.Tensor' inputs to the
            `FusionDefinition` `fd`.
        weight: If given, multiply normed output by this `Tensor`. It should be
            one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
            zero-dimensional otherwise. It will be broadcast along all other
            dimensions.
        bias: If given, add this `Tensor` to normed output. It should be
            one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
            zero-dimensional otherwise. It will be broadcast along all other
            dimensions.
        running_mean: If given, a running mean estimate that will be modified
            in place.
        running_var: If given, a running variance estimate that will be
            modified in place.
        use_input_stats: Whether to compute the stats of this batch or to
            _only_ use the provided running_mean and running_var.
        channels_last: Whether channels are in position -1 (`True`) or 1
            (`False`).
        x_datatype: :class:'DataType' of input :class:'Tensor' `x`
        extent: Size of the input.
        stat_axes: A list of `NamedAxis` objects indicating a combination of
            axes with which to index the computed statistics. This can be used
            to implement multiple types of normalization layers, since most of
            those differ only in which axes are reduced over.
    Returns:
        The normalized output, as well as mean and 1/std. Note that
        `fd.add_output` is _not_ called by this function.
    """
    assert not (
        (running_var is None) ^ (running_mean is None)
    ), "Iff running mean or var is given, the other should be"

    batch_dim = 0
    num_dims = len(extent)
    channel_dim = num_dims - 1 if channels_last else 1

    stat_dims = []
    # Running stats will be kept possibly for channel but never by instance, so
    # we will reduce along batch_dim before updating running stats.
    stat_dims_nobatch = []
    num_stats = 1
    if NamedAxis.BATCH in stat_axes:
        stat_dims.append(batch_dim)
        num_stats *= extent[batch_dim]
    if NamedAxis.CHANNEL in stat_axes:
        stat_dims.append(channel_dim)
        stat_dims_nobatch.append(channel_dim)
        num_stats *= extent[channel_dim]
    x_reduction_axes = [ax for ax in range(num_dims) if ax not in stat_dims]
    num_features = extent.numel() // num_stats

    batch_size = fd.define_constant(extent[batch_dim])

    mean = fd.ops.broadcast_in_dim(mean, extent, [batch_dim, channel_dim])

    norm = fd.define_constant(1.0 / num_features)
    grad_output_sum = fd.ops.sum(grad_output, x_reduction_axes)
    dot_p = fd.ops.sum(
        fd.ops.mul(
            grad_output,
            fd.ops.sub(x, mean),
        ),
        x_reduction_axes,
    )
    grad_mean = fd.ops.broadcast_in_dim(
        fd.ops.mul(grad_output_sum, norm),
        extent,
        [batch_dim, channel_dim],
    )
    proj_scale = fd.ops.broadcast_in_dim(
        fd.ops.mul(
            fd.ops.mul(dot_p, norm),
            fd.ops.mul(invstd, invstd),
        ),
        extent,
        [batch_dim, channel_dim],
    )

    invstd_bcast = fd.ops.broadcast_in_dim(
        invstd,
        extent,
        [batch_dim, channel_dim],
    )
    grad_scale = (
        invstd_bcast
        if weight is None
        else fd.ops.mul(
            invstd_bcast,
            fd.ops.broadcast_in_dim(weight, extent, [0]),
        )
    )
    if use_input_stats:
        proj = fd.ops.mul(fd.ops.sub(x, mean), proj_scale)
        grad_input = fd.ops.mul(
            fd.ops.sub(
                fd.ops.sub(grad_output, proj),
                grad_mean,
            ),
            grad_scale,
        )
    else:
        grad_input = fd.ops.mul(grad_output, grad_scale)

    if weight is not None:
        grad_weight = fd.ops.mul(dot_p, invstd)
        grad_weight_reduced = fd.ops.sum(grad_weight, [0])
    else:
        grad_weight_reduced = None
    if bias is not None:
        grad_bias = grad_output_sum
        grad_bias_reduced = fd.ops.sum(grad_bias, [0])
    else:
        grad_bias_reduced = None

    return grad_input, grad_weight_reduced, grad_bias_reduced


# The fusion caches for forward and backward evaluation map "keys" in the form
# of tuples of parameters, to Fusion objects. These Fusion objects hold
# FusionExecutorCache objects under the hood, which are able to repeatedly
# execute a Fusion without redefining or rescheduling them.
_forward_fusion_cache: Dict[
    Tuple[
        Tuple[NamedAxis, ...],
        DataType,
        DataType,
        DataType,
        DataType,
        int,
        bool,
        bool,
        bool,
        bool,
    ],
    Fusion,
] = {}
_backward_fusion_cache: Dict[
    Tuple[
        Tuple[NamedAxis, ...],
        DataType,
        DataType,
        DataType,
        DataType,
        int,
        bool,
        bool,
        bool,
        bool,
    ],
    Fusion,
] = {}


class NormNVFuserFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # contexts are actually objects of the type we are currently defining
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        running_mean: Optional[torch.Tensor],
        running_var: Optional[torch.Tensor],
        use_input_stats: bool,
        momentum: float,
        eps: float,
        unbiased: bool,
        stat_axes: List[NamedAxis],
    ) -> torch.Tensor:
        channels_last = x.is_contiguous(
            memory_format=torch.channels_last
        ) or x.is_contiguous(memory_format=torch.channels_last_3d)
        xorig = x
        if channels_last:
            order = [0] + [i for i in range(2, len(x.shape))] + [1]
            x = x.permute(order)
        assert x.is_contiguous()

        x_datatype = torch2datatype(x.dtype)

        # The cache key for this Fusion definition. Misses will trigger redefinition and rescheduling.
        key = (
            tuple(stat_axes),  # convert to tuple to make hashable
            x_datatype,
            None if weight is None else torch2datatype(weight.dtype),
            None if bias is None else torch2datatype(bias.dtype),
            None if running_mean is None else torch2datatype(running_mean.dtype),
            x.ndim,
            channels_last,
            running_mean is not None,
            weight is not None,
            bias is not None,
        )
        fs = _forward_fusion_cache.get(key)
        if fs is None:  # cache miss
            fs = Fusion()
            with FusionDefinition(fs) as fd:
                tv_x = fd.define_tensor(x.ndim, torch2datatype(x.dtype))
                inputs = [x]
                if weight is not None:
                    tv_weight = fd.define_tensor(
                        weight.ndim, torch2datatype(weight.dtype)
                    )
                    inputs.append(weight)
                else:
                    tv_weight = None

                if bias is not None:
                    tv_bias = fd.define_tensor(bias.ndim, torch2datatype(bias.dtype))
                    inputs.append(bias)
                else:
                    tv_bias = None

                if running_mean is None:
                    tv_running_mean = None
                    tv_running_var = None
                else:
                    assert running_var is not None
                    tv_running_mean = fd.define_tensor(
                        running_mean.ndim, torch2datatype(running_mean.dtype)
                    )
                    tv_running_var = fd.define_tensor(
                        running_var.ndim, torch2datatype(running_var.dtype)
                    )
                    inputs.extend([running_mean, running_var])
                    if running_mean.dtype in [torch.half, torch.bfloat16]:
                        tv_running_mean = fd.ops.cast(tv_running_mean, DataType.Float)
                    if running_var.dtype in [torch.half, torch.bfloat16]:
                        tv_running_var = fd.ops.cast(tv_running_var, DataType.Float)

                s_momentum = fd.define_scalar(DataType.Double)
                s_eps = fd.define_scalar(DataType.Double)
                inputs.extend([momentum, eps])

                # cast inputs if necessary
                if x_datatype in [DataType.Half, DataType.BFloat16]:
                    tv_x = fd.ops.cast(tv_x, DataType.Float)
                if weight is not None and weight.dtype in [torch.half, torch.bfloat16]:
                    tv_weight = fd.ops.cast(tv_weight, DataType.Float)
                if bias is not None and bias.dtype in [torch.half, torch.bfloat16]:
                    tv_bias = fd.ops.cast(tv_bias, DataType.Float)

                out, mean, invstd = norm_fusion_forward(
                    fd,
                    tv_x,
                    tv_weight,
                    tv_bias,
                    tv_running_mean,
                    tv_running_var,
                    s_eps,
                    use_input_stats,
                    s_momentum,
                    channels_last,
                    x_datatype=x_datatype,
                    extent=x.shape,
                    unbiased=unbiased,
                    stat_axes=stat_axes,
                )

                if x_datatype in [DataType.Half, DataType.BFloat16]:
                    out = fd.ops.cast(out, x_datatype)
                    mean = fd.ops.cast(mean, x_datatype)
                    invstd = fd.ops.cast(invstd, x_datatype)

                fd.add_output(out)
                fd.add_output(mean)
                fd.add_output(invstd)

            _forward_fusion_cache[key] = fs
        else:  # cache hit. define inputs to match above
            inputs = [x]
            if weight is not None:
                inputs.append(weight)
            if bias is not None:
                inputs.append(bias)
            if running_mean is not None:
                assert running_var is not None
                inputs.extend([running_mean, running_var])
            inputs.extend([momentum, eps])

        out, mean, invstd = fs.execute(inputs)

        ctx.stat_axes = stat_axes
        ctx.use_input_stats = use_input_stats
        ctx.channels_last = channels_last
        # saving for backward in "explicit channels-last format"
        ctx.save_for_backward(x, weight, bias, running_mean, running_var, mean, invstd)
        if channels_last:
            order = [0, len(x.shape) - 1] + [i for i in range(1, len(x.shape) - 1)]
            out = out.permute(order)
            if len(out.shape) == 4:
                assert out.is_contiguous(memory_format=torch.channels_last)
                assert xorig.is_contiguous(memory_format=torch.channels_last)
            elif len(out.shape) == 5:
                assert out.is_contiguous(memory_format=torch.channels_last_3d)
                assert xorig.is_contiguous(memory_format=torch.channels_last_3d)
            else:
                assert False, "unhandled channels_last format variation in forward"
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None
    ]:
        """
        Instance norm backward using NVFuser
        """
        if ctx.channels_last:
            order = [0] + [i for i in range(2, len(grad_output.shape))] + [1]
            grad_output = grad_output.permute(order)
        # input was saved in "explicit channels-last format"
        assert ctx.saved_tensors[0].is_contiguous()
        grad_output = grad_output.contiguous()
        x, weight, bias, running_mean, running_var, mean, invstd = ctx.saved_tensors

        # The cache key for this Fusion definition. Misses will trigger redefinition and rescheduling.
        key = (
            tuple(ctx.stat_axes),  # convert to tuple to make hashable
            torch2datatype(x.dtype),
            None if weight is None else torch2datatype(weight.dtype),
            None if bias is None else torch2datatype(bias.dtype),
            None if running_mean is None else torch2datatype(running_mean.dtype),
            x.ndim,
            ctx.channels_last,
            running_mean is not None,
            weight is not None,
            bias is not None,
        )
        fs = _backward_fusion_cache.get(key)
        if fs is None:  # cache miss
            fs = Fusion()
            with FusionDefinition(fs) as fd:
                tv_x = fd.define_tensor(x.ndim, torch2datatype(x.dtype))
                inputs = [x]
                if weight is not None:
                    tv_weight = fd.define_tensor(
                        weight.ndim, torch2datatype(weight.dtype)
                    )
                    inputs.append(weight)
                else:
                    tv_weight = None
                if bias is not None:
                    tv_bias = fd.define_tensor(bias.ndim, torch2datatype(bias.dtype))
                    inputs.append(bias)
                else:
                    tv_bias = None
                if running_mean is not None:
                    tv_running_mean = fd.define_tensor(
                        running_mean.ndim, torch2datatype(running_mean.dtype)
                    )
                    inputs.append(running_mean)
                else:
                    tv_running_mean = None
                if running_var is not None:
                    tv_running_var = fd.define_tensor(
                        running_var.ndim, torch2datatype(running_var.dtype)
                    )
                    inputs.append(running_var)
                else:
                    tv_running_var = None

                tv_mean = fd.define_tensor(mean.ndim, torch2datatype(mean.dtype))
                inputs.append(mean)
                tv_invstd = fd.define_tensor(invstd.ndim, torch2datatype(invstd.dtype))
                inputs.append(invstd)

                tv_grad_output = fd.define_tensor(grad_output.ndim)
                inputs.append(grad_output)

                x_datatype = torch2datatype(x.dtype)

                grad_input, grad_weight, grad_bias = norm_fusion_backward(
                    fd,
                    tv_x,
                    tv_grad_output,
                    tv_mean,
                    tv_invstd,
                    inputs,
                    tv_weight,
                    tv_bias,
                    tv_running_mean,
                    tv_running_var,
                    ctx.use_input_stats,
                    ctx.channels_last,
                    x_datatype=x_datatype,
                    extent=x.shape,
                    stat_axes=ctx.stat_axes,
                )

                if x_datatype in [DataType.Half, DataType.BFloat16]:
                    grad_input = fd.ops.cast(grad_input, x_datatype)
                fd.add_output(grad_input)

                if weight is not None:
                    if x_datatype in [DataType.Half, DataType.BFloat16]:
                        grad_weight = fd.ops.cast(grad_weight, x_datatype)
                    fd.add_output(grad_weight)

                if bias is not None:
                    if x_datatype in [DataType.Half, DataType.BFloat16]:
                        grad_bias = fd.ops.cast(grad_bias, x_datatype)
                    else:
                        fd.add_output(grad_bias)

            _backward_fusion_cache[key] = fs
        else:  # cache hit. define inputs to match above
            inputs = [x]
            if weight is not None:
                inputs.append(weight)
            if bias is not None:
                inputs.append(bias)
            if running_mean is not None:
                assert running_var is not None
                inputs.extend([running_mean, running_var])
            inputs.extend([mean, invstd, grad_output])

        res = fs.execute(inputs)
        grad_input = res[0]
        c = 1
        if weight is not None:
            grad_weight = res[c]
            c += 1
        else:
            grad_weight = None
        if bias is not None:
            grad_bias = res[c]
            c += 1
        else:
            grad_bias = None

        if ctx.channels_last:
            order = [0, len(grad_input.shape) - 1] + [
                i for i in range(1, len(grad_input.shape) - 1)
            ]
            grad_input = grad_input.permute(order)
            if len(grad_input.shape) == 4:
                assert grad_input.is_contiguous(memory_format=torch.channels_last)
            elif len(grad_input.shape) == 5:
                assert grad_input.is_contiguous(memory_format=torch.channels_last_3d)
            else:
                assert False, "unhandled channels_last format variation in backward"
        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _NormNVFuserBase(_NormBase):
    stat_axes = None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ("running_mean", "running_var"):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    "Unexpected running stats buffer(s) {names} for {klass} "
                    "with track_running_stats=False. If state_dict is a "
                    "checkpoint saved before 0.4.0, this may be expected "
                    "because {klass} does not track running stats by default "
                    "since 0.4.0. Please remove these keys from state_dict. If "
                    "the running stats are actually needed, instead set "
                    "track_running_stats=True in {klass} to enable them. See "
                    "the documentation of {klass} for details.".format(
                        names=" and ".join(
                            '"{}"'.format(k) for k in running_stats_keys
                        ),
                        klass=self.__class__.__name__,
                    )
                )
                for key in running_stats_keys:
                    state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: Tensor) -> Tensor:
        assert input.is_cuda, "NVFuser InstanceNorm is CUDA only"
        self._check_input_dim(input)
        out = NormNVFuserFunction.apply(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
            False,
            self.stat_axes,
        )
        return out


class _InstanceNormNVFuser(_NormNVFuserBase):
    stat_axes = [NamedAxis.BATCH, NamedAxis.CHANNEL]


class _BatchNormNVFuser(_NormNVFuserBase):
    stat_axes = [NamedAxis.CHANNEL]


class _LayerNormNVFuser(_NormNVFuserBase):
    stat_axes = [NamedAxis.BATCH]


class InstanceNorm1dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError("expected 3D input (got {}D input)".format(input.dim()))


class InstanceNorm2dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class InstanceNorm3dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
