from typing import Any, Optional, Tuple

import torch
from torch.nn.modules.batchnorm import _NormBase

from nvfuser._C import DataType, Fusion, FusionDefinition, Scalar, Tensor


__all__ = ["InstanceNormNVFuserFunction", "InstanceNorm3dNVFuser"]


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


def instance_norm(
    fd: FusionDefinition,
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    use_input_stats: bool,
    momentum: Scalar,
    eps: Scalar,
    channels_last: bool,
    unbiased: bool,
    extent: torch.Size,
    x_datatype: DataType,
) -> Tensor:
    """Compute instance norm layer forward for arbitrary dimensional input.

    This is a translation of `instance_norm` in NVFuser [^1] which is not
    exposed currently in the Python frontend

    [^1]: https://github.com/csarofeen/pytorch/blob/devel/third_party/nvfuser/csrc/ops/normalization.cpp#L710
    """
    assert not (
        (running_var is None) ^ (running_mean is None)
    ), "Iff running mean or var is given, the other should be"

    kBatchDim = 0
    kNumberOfDims = len(extent)
    kChannelsDim = kNumberOfDims - 1 if channels_last else 1

    num_features = extent.numel() // (extent[kBatchDim] * extent[kChannelsDim])

    x_reduction_axes = [
        axis for axis in range(kNumberOfDims) if axis not in [kBatchDim, kChannelsDim]
    ]
    B = fd.define_constant(extent[kBatchDim])

    y = None
    mean = None
    invstd = None
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

            new_mean_sum = fd.ops.sum(new_mean_hat, [kBatchDim])
            rB = fd.ops.reciprocal(B)
            new_mean_channels_only = fd.ops.mul(new_mean_sum, rB)
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

            new_var_sum = fd.ops.sum(new_var_hat, [kBatchDim])
            new_var_channels_only = fd.ops.mul(new_var_sum, rB)
            if x_datatype in [DataType.Half, DataType.BFloat16]:
                new_var_channels_only = fd.ops.cast(new_var_channels_only, x_datatype)
            fd.add_output(new_var_channels_only, alias_input=running_var)

        mean = x_mean
        mean_bcast = fd.ops.broadcast_in_dim(mean, extent, [kBatchDim, kChannelsDim])
        x_sub_mean = fd.ops.sub(x, mean_bcast)

        var_eps = fd.ops.add(x_var, eps)
        invstd = fd.ops.rsqrt(var_eps)
        invstd_bcast = fd.ops.broadcast_in_dim(
            invstd, extent, [kBatchDim, kChannelsDim]
        )

        y = fd.ops.mul(x_sub_mean, invstd_bcast)

    else:  # This is inference mode with running stats
        r_mean_bcast = fd.ops.broadcast_in_dim(running_mean, extent, [kChannelsDim])
        x_sub_mean = fd.ops.sub(x, r_mean_bcast)

        var_eps = fd.ops.add(running_var, eps)
        invstd = fd.ops.rsqrt(var_eps)
        invstd_bcast = fd.ops.broadcast_in_dim(invstd, extent, [kChannelsDim])

        mean = running_mean
        y = fd.ops.mul(x_sub_mean, invstd_bcast)

    if weight is not None:
        weight_bcast = fd.ops.broadcast_in_dim(weight, extent, [kChannelsDim])
        y = fd.ops.mul(y, weight_bcast)
    if bias is not None:
        bias_bcast = fd.ops.broadcast_in_dim(bias, extent, [kChannelsDim])
        y = fd.ops.add(y, bias_bcast)

    return y, mean, invstd


class InstanceNormNVFuserFunction(torch.autograd.Function):
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
        unbiased: bool = False,
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

        # execute fusion using Python API. Will be cached automatically
        fs = Fusion()
        with FusionDefinition(fs) as fd:
            tv_x = fd.define_tensor(x.ndim, torch2datatype(x.dtype))
            inputs = [x]
            if weight is not None:
                assert bias is not None
                tv_weight = fd.define_tensor(weight.ndim, torch2datatype(weight.dtype))
                tv_bias = fd.define_tensor(bias.ndim, torch2datatype(bias.dtype))
                inputs.extend([weight, bias])
            else:
                tv_weight = None
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

            out, mean, invstd = instance_norm(
                fd,
                tv_x,
                tv_weight,
                tv_bias,
                tv_running_mean,
                tv_running_var,
                use_input_stats,
                s_momentum,
                s_eps,
                channels_last,
                unbiased=unbiased,
                extent=x.shape,
                x_datatype=x_datatype,
            )

            if x_datatype in [DataType.Half, DataType.BFloat16]:
                out = fd.ops.cast(out, x_datatype)
                mean = fd.ops.cast(mean, x_datatype)
                invstd = fd.ops.cast(invstd, x_datatype)

            fd.add_output(out)
            fd.add_output(mean)
            fd.add_output(invstd)

        out, mean, invstd = fs.execute(inputs)

        ctx.use_input_stats = use_input_stats
        ctx.eps = eps
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None,]:
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

        kBatchDim = 0
        kNumberOfDims = x.ndim
        kChannelsDim = kNumberOfDims - 1 if ctx.channels_last else 1
        kTraining = ctx.use_input_stats
        reduction_axes = [
            axis
            for axis in range(kNumberOfDims)
            if axis not in [kBatchDim, kChannelsDim]
        ]

        fs = Fusion()
        with FusionDefinition(fs) as fd:
            tv_x = fd.define_tensor(x.ndim, torch2datatype(x.dtype))
            inputs = [x]
            if weight is not None:
                tv_weight = fd.define_tensor(weight.ndim, torch2datatype(weight.dtype))
                inputs.extend([weight])
            else:
                tv_weight = None

            tv_grad_output = fd.define_tensor(grad_output.ndim)
            inputs.append(grad_output)

            if kTraining:
                assert mean is not None and invstd is not None
                tv_mean = fd.define_tensor(mean.ndim)
                inputs.append(mean)
                tv_invstd = fd.define_tensor(invstd.ndim)
                inputs.append(invstd)
            else:
                tv_running_mean = fd.define_tensor(running_mean.ndim)
                inputs.append(running_mean)
                tv_running_var = fd.define_tensor(running_var.ndim)
                inputs.append(running_var)
                c_eps = fd.define_constant(DataType.Double, ctx.eps)

                tv_mean = tv_running_mean
                tv_invstd = fd.ops.rsqrt(fd.ops.add(tv_running_var, c_eps))

            tv_mean = fd.ops.broadcast_in_dim(
                tv_mean, x.shape, [kBatchDim, kChannelsDim]
            )

            num_features = x.numel() // (x.shape[kBatchDim] * x.shape[kChannelsDim])

            norm = fd.define_constant(1.0 / num_features)
            grad_output_sum = fd.ops.sum(tv_grad_output, reduction_axes)
            dot_p = fd.ops.sum(
                fd.ops.mul(
                    tv_grad_output,
                    fd.ops.sub(tv_x, tv_mean),
                ),
                reduction_axes,
            )
            grad_mean = fd.ops.broadcast_in_dim(
                fd.ops.mul(grad_output_sum, norm),
                x.shape,
                [kBatchDim, kChannelsDim],
            )
            proj_scale = fd.ops.broadcast_in_dim(
                fd.ops.mul(
                    fd.ops.mul(dot_p, norm),
                    fd.ops.mul(tv_invstd, tv_invstd),
                ),
                x.shape,
                [kBatchDim, kChannelsDim],
            )

            invstd_bcast = fd.ops.broadcast_in_dim(
                tv_invstd,
                x.shape,
                [kBatchDim, kChannelsDim],
            )
            grad_scale = (
                invstd_bcast
                if weight is None
                else fd.ops.mul(
                    invstd_bcast,
                    fd.ops.broadcast_in_dim(tv_weight, x.shape, [0]),
                )
            )
            if kTraining:
                proj = fd.ops.mul(fd.ops.sub(tv_x, tv_mean), proj_scale)
                grad_input = fd.ops.mul(
                    fd.ops.sub(
                        fd.ops.sub(tv_grad_output, proj),
                        grad_mean,
                    ),
                    grad_scale,
                )
            else:
                grad_input = fd.ops.mul(tv_grad_output, grad_scale)

            x_datatype = torch2datatype(x.dtype)
            if x_datatype in [DataType.Half, DataType.BFloat16]:
                fd.add_output(fd.ops.cast(grad_input, x_datatype))
            else:
                fd.add_output(grad_input)

            if weight is not None:
                grad_weight = fd.ops.mul(dot_p, tv_invstd)
                grad_weight_reduced = fd.ops.sum(grad_weight, [0])
                if x_datatype in [DataType.Half, DataType.BFloat16]:
                    fd.add_output(fd.ops.cast(grad_weight_reduced, x_datatype))
                else:
                    fd.add_output(grad_weight_reduced)

            if bias is not None:
                grad_bias = grad_output_sum
                grad_bias_reduced = fd.ops.sum(grad_bias, [0])
                if x_datatype in [DataType.Half, DataType.BFloat16]:
                    fd.add_output(fd.ops.cast(grad_bias_reduced, x_datatype))
                else:
                    fd.add_output(grad_bias_reduced)

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
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class _InstanceNormNVFuser(_NormBase):
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
        super(_InstanceNormNVFuser, self).__init__(
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

        super(_InstanceNormNVFuser, self)._load_from_state_dict(
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
        out = InstanceNormNVFuserFunction.apply(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )
        return out


class InstanceNorm3dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
