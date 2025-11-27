"""Customized CuDNN frontend layer norm.

Please refer to:

* https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/20_layernorm.ipynb
"""

from __future__ import annotations

import math

import cudnn
import torch

__all__ = ["get_cudnn_manager"]


class CuDNNManager:
    """CuDNN fronted context manager.

    Notice: CuDNN handle must be created after distributed process group initialization.
    """

    def __init__(self) -> None:
        self._handle = cudnn.create_handle()
        self._cudnn_stream = torch.cuda.Stream()
        self.reset_stream()

    def __del__(self) -> None:
        if cudnn is not None and hasattr(cudnn, "destroy_handle"):
            cudnn.destroy_handle(self._handle)

    def __enter__(self) -> CuDNNManager:
        self._torch_stream = torch.cuda.current_stream()
        self._cudnn_stream.wait_stream(self._torch_stream)
        torch.cuda.set_stream(self._cudnn_stream)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object | None,
    ) -> None:
        self._torch_stream.wait_stream(self._cudnn_stream)
        torch.cuda.set_stream(self._torch_stream)
        del self._torch_stream

    def set_stream(self, stream: torch.cuda.Stream) -> None:
        cudnn.set_stream(stream=stream.cuda_stream, handle=self._handle)

    def reset_stream(self) -> None:
        cudnn.set_stream(stream=self._cudnn_stream.cuda_stream, handle=self._handle)

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def stream(self) -> torch.cuda.Stream:
        return self._cudnn_stream


_global_cudnn_manager: CuDNNManager | None = None


def get_cudnn_manager() -> CuDNNManager:
    """Get the CuDNN front-end context manager.

    Returns:
        CuDNNManager: Global CuDNN manager.
    """
    global _global_cudnn_manager
    if _global_cudnn_manager is None:
        _global_cudnn_manager = CuDNNManager()
    return _global_cudnn_manager


class LayerNormGraphFactory:
    """cuDNN front-end layer norm graph factory.

    cuDNN layer norm constraints:

    * All tensors are 4-dimensional;
    * `x` and `y` have the same layout in the graph;
    """

    _graphs: dict = {}
    _symbols: dict = {}
    _TORCH2CUDNN: dict = {
        torch.bool: cudnn.data_type.BOOLEAN,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float16: cudnn.data_type.HALF,
        torch.float32: cudnn.data_type.FLOAT,
        torch.uint8: cudnn.data_type.UINT8,
    }

    @classmethod
    def get_forward_graph(
        cls: type[LayerNormGraphFactory],
        m: int,
        n: int,
        xdtype: torch.dtype,
        wdtype: torch.dtype,
    ) -> tuple[cudnn._compiled_module.pygraph, tuple]:
        key = m, n, xdtype, wdtype, "FORWARD"
        if key in cls._graphs:
            if key not in cls._symbols:
                raise RuntimeError(
                    f"Symbolic tensor was not constructed for layer-norm forward graph with input "
                    f"shape {(m, n)} and data type {(xdtype, wdtype)}",
                )
            return cls._graphs[key], cls._symbols[key]

        cudnn_manager: CuDNNManager = get_cudnn_manager()
        graph = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_manager.handle,
        )
        x_sym = graph.tensor(
            name="x_sym",
            dim=(m, n, 1, 1),
            stride=(n, 1, n, n),  # Simulate the channel-last format.
            data_type=cls._TORCH2CUDNN[xdtype],
        )
        scale_sym = graph.tensor(
            name="scale_sym",
            dim=(1, n, 1, 1),
            stride=(n, 1, n, n),
            data_type=cls._TORCH2CUDNN[wdtype],
        )
        bias_sym = graph.tensor(
            name="bias_sym",
            dim=(1, n, 1, 1),
            stride=(n, 1, n, n),
            data_type=cls._TORCH2CUDNN[wdtype],
        )
        eps_sym = graph.tensor(
            name="eps_sym",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            is_pass_by_value=True,
            data_type=cudnn.data_type.FLOAT,
        )

        y_sym, x_mean_sym, x_invstd_sym = graph.layernorm(
            name=f"layer-norm-forward-{key}",
            norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
            input=x_sym,
            scale=scale_sym,
            bias=bias_sym,
            epsilon=eps_sym,
        )

        y_sym.set_output(True).set_data_type(cls._TORCH2CUDNN[xdtype])
        x_mean_sym.set_output(True).set_data_type(cls._TORCH2CUDNN[torch.float32])
        x_invstd_sym.set_output(True).set_data_type(cls._TORCH2CUDNN[torch.float32])

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)  # ALL
        symbols = (
            x_sym,
            scale_sym,
            bias_sym,
            eps_sym,
            y_sym,
            x_mean_sym,
            x_invstd_sym,
        )

        cls._graphs[key] = graph
        cls._symbols[key] = symbols

        return graph, symbols

    @classmethod
    def get_backward_graph(
        cls: type[LayerNormGraphFactory],
        m: int,
        n: int,
        xdtype: torch.dtype,
        wdtype: torch.dtype,
    ) -> tuple[cudnn._compiled_module.pygraph, tuple]:
        key = m, n, xdtype, wdtype, "BACKWARD"
        if key in cls._graphs:
            if key not in cls._symbols:
                raise RuntimeError(
                    f"Symbolic tensor was not constructed for layer-norm backward "
                    f"graph with input shape {(m, n)} and data type {(xdtype, wdtype)}",
                )
            return cls._graphs[key], cls._symbols[key]

        cudnn_manager: CuDNNManager = get_cudnn_manager()
        graph = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_manager.handle,
        )
        x_sym = graph.tensor(
            name="x_sym",
            dim=(m, n, 1, 1),
            stride=(n, 1, n, n),  # Simulate the channel-last format.
            data_type=cls._TORCH2CUDNN[xdtype],
        )
        d_y_sym = graph.tensor(
            name="d_y_sym",
            dim=(m, n, 1, 1),
            stride=(n, 1, n, n),  # Simulate the channel-last format.
            data_type=cls._TORCH2CUDNN[xdtype],
        )
        scale_sym = graph.tensor(
            name="scale_sym",
            dim=(1, n, 1, 1),
            stride=(n, 1, n, n),
            data_type=cls._TORCH2CUDNN[wdtype],
        )
        x_mean_sym = graph.tensor(
            name="x_mean_sym",
            dim=(m, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        x_invstd_sym = graph.tensor(
            name="x_invstd_sym",
            dim=(m, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        d_x_sym, d_scale_sym, d_bias_sym = graph.layernorm_backward(
            name=f"layer-norm-backward-{key}",
            grad=d_y_sym,
            input=x_sym,
            scale=scale_sym,
            mean=x_mean_sym,
            inv_variance=x_invstd_sym,
        )

        d_x_sym.set_output(True).set_data_type(cls._TORCH2CUDNN[xdtype])
        d_scale_sym.set_output(True).set_data_type(cls._TORCH2CUDNN[wdtype])
        d_bias_sym.set_output(True).set_data_type(cls._TORCH2CUDNN[wdtype])

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)  # ALL
        symbols = (
            x_sym,
            d_y_sym,
            scale_sym,
            x_mean_sym,
            x_invstd_sym,
            d_x_sym,
            d_scale_sym,
            d_bias_sym,
        )

        cls._graphs[key] = graph
        cls._symbols[key] = symbols

        return graph, symbols


@torch.library.custom_op("cudnn::layer_norm", mutates_args=(), device_types="cuda")
def layer_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # PyTorch LayerNorm:
    #   * Shape (N, S, H), normalized_shape (H,);
    #   * Shape (N, C, H, W), normalized_shape (C, H, W);
    # cuDNN LayerNorm expects shape (M, N, 1, 1) and normalized_shape (1, N, 1, 1)
    if tuple(x.shape[-len(normalized_shape) :]) != tuple(normalized_shape):  # noqa: E203
        raise ValueError(
            f"CuDNN LayerNorm expects `x.shape[{-len(normalized_shape)}:]` equals to "
            f"`normalized_shape`, but got:\n    {x.shape=}, {normalized_shape=}",
        )
    assert weight.dtype == bias.dtype
    assert x.is_contiguous()

    stream = torch.cuda.current_stream()
    cudnn_manager: CuDNNManager = get_cudnn_manager()
    cudnn_manager.set_stream(stream)

    xdtype, wdtype, device = x.dtype, weight.dtype, x.device
    m, n = math.prod(x.shape[: -len(normalized_shape)]), math.prod(normalized_shape)
    (
        forward_graph,
        (
            x_sym,
            scale_sym,
            bias_sym,
            eps_sym,
            y_sym,
            x_mean_sym,
            x_invstd_sym,
        ),
    ) = LayerNormGraphFactory.get_forward_graph(m, n, xdtype, wdtype)

    x_contiguous = x.reshape(m, n, 1, 1)  # NOTE: x could be noncontiguous.
    weight = weight.view(1, n, 1, 1)
    bias = bias.view(1, n, 1, 1)
    eps_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    y = torch.empty_like(x_contiguous)
    x_mean = torch.empty(m, 1, 1, 1, dtype=torch.float32, device=device)
    x_invstd = torch.empty(m, 1, 1, 1, dtype=torch.float32, device=device)
    workspace = torch.empty(
        forward_graph.get_workspace_size(),
        dtype=torch.uint8,
        device=device,
    )

    forward_graph.execute(
        {
            x_sym: x_contiguous.detach(),
            scale_sym: weight.detach(),
            bias_sym: bias.detach(),
            eps_sym: eps_cpu.detach(),
            y_sym: y.detach(),
            x_mean_sym: x_mean.detach(),
            x_invstd_sym: x_invstd.detach(),
        },
        workspace,
    )
    y = y.view(x.shape)

    return y, x_mean, x_invstd


@layer_norm.register_fake
def layer_norm_fake(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = math.prod(x.shape[: -len(normalized_shape)])

    y = torch.empty_like(x)
    x_mean = torch.empty(m, 1, 1, 1, dtype=torch.float32, device=x.device)
    x_invstd = torch.empty(m, 1, 1, 1, dtype=torch.float32, device=x.device)
    return y, x_mean, x_invstd


@torch.library.custom_op(
    "cudnn::layer_norm_backward",
    mutates_args=(),
    device_types="cuda",
)
def layer_norm_backward(
    d_y: torch.Tensor,
    x_mean: torch.Tensor,
    x_invstd: torch.Tensor,
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xdtype, wdtype, device = d_y.dtype, weight.dtype, d_y.device
    m, n = math.prod(x.shape[: -len(normalized_shape)]), math.prod(normalized_shape)

    stream = torch.cuda.current_stream()
    cudnn_manager: CuDNNManager = get_cudnn_manager()
    cudnn_manager.set_stream(stream)

    (
        backward_graph,
        (
            x_sym,
            d_y_sym,
            scale_sym,
            x_mean_sym,
            x_invstd_sym,
            d_x_sym,
            d_scale_sym,
            d_bias_sym,
        ),
    ) = LayerNormGraphFactory.get_backward_graph(m, n, xdtype, wdtype)

    d_y_contiguous = d_y.reshape(m, n, 1, 1)  # NOTE: d_y could also be noncontiguous.
    d_x = torch.empty_like(x)
    d_weight = torch.empty_like(weight)
    d_bias = torch.empty_like(bias)
    workspace = torch.empty(
        backward_graph.get_workspace_size(),
        dtype=torch.uint8,
        device=device,
    )

    backward_graph.execute(
        {
            x_sym: x.detach(),
            d_y_sym: d_y_contiguous.detach(),
            scale_sym: weight.detach(),
            x_mean_sym: x_mean.detach(),
            x_invstd_sym: x_invstd.detach(),
            d_x_sym: d_x.detach(),
            d_scale_sym: d_weight.detach(),
            d_bias_sym: d_bias.detach(),
        },
        workspace,
    )

    return d_x, d_weight, d_bias


@layer_norm_backward.register_fake
def layer_norm_backward_fake(
    d_y: torch.Tensor,
    x_mean: torch.Tensor,
    x_invstd: torch.Tensor,
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_x = torch.empty_like(x)
    d_weight = torch.empty_like(weight)
    d_bias = torch.empty_like(bias)
    return d_x, d_weight, d_bias


def layer_norm_setup_context(
    ctx: torch.autograd.FunctionCtx,
    inputs: tuple,
    output: tuple,
) -> torch.Tensor:
    x, normalized_shape, weight, bias, eps = inputs
    y, x_mean, x_invstd = output

    ctx.save_for_backward(x, weight, bias, x_mean, x_invstd)
    ctx.normalized_shape = normalized_shape

    return y


def layer_norm_backward_wrapper(
    ctx: torch.autograd.FunctionCtx,
    d_y: torch.Tensor,
    d_x_mean: torch.Tensor,
    d_x_invstd: torch.Tensor,
) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, None]:
    x, weight, bias, x_mean, x_invstd = ctx.saved_tensors
    normalized_shape = ctx.normalized_shape

    d_x, d_weight, d_bias = layer_norm_backward(
        d_y,
        x_mean,
        x_invstd,
        x,
        normalized_shape,
        weight,
        bias,
    )

    return d_x, None, d_weight, d_bias, None


torch.library.register_autograd(
    "cudnn::layer_norm",
    layer_norm_backward_wrapper,
    setup_context=layer_norm_setup_context,
)
