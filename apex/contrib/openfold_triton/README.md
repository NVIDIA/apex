# OpenFold triton kernels

This subpackage is a collection of Triton kernels written specifically for the OpenFold model architecture initial training mode.

To use this subpackage, you must install additional dependencies:

```bash
pip install einops
```

The following sections list all main features and show how to use them.

## Multi-Head Attention

```python
import apex.contrib.openfold_triton.mha as mha
from apex.contrib.openfold_triton import AttnBiasJIT, AttnNoBiasJIT, AttnTri, CanSchTriMHA

# Integration with Attention module:
class SelfAttentionWithGate(nn.Module):
    # ...

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.chunk_size is None:
            if mha.is_enabled() and CanSchTriMHA(
                list(query.shape),
                bias is not None,
                inf=self.inf,
                training=self.training,
            ):
                if mask is not None:
                    mask = mask.contiguous()
                if bias is not None:
                    bias = bias.contiguous()
                return AttnTri(
                    query, key, value, mask, bias, self.inf, torch.is_grad_enabled()
                )
            elif mha.is_enabled() and bias is not None and self.training:
                return AttnBiasJIT(query, key, value, mask, bias, self.inf)
            elif mha.is_enabled() and bias is None and self.training:
                return AttnNoBiasJIT(query, key, value, mask, self.inf)

# Switch on/off MHA dynamically at runtime via:
mha.enable()
mha.disable()

```

## LayerNorm

```python
from apex.contrib.openfold_triton import LayerNormSmallShapeOptImpl

# Integration with LayerNorm module:
class LayerNorm(nn.Module):
    # ...

    def _should_use_triton_kernels(self, x: torch.Tensor) -> bool:
        ln_triton_shapes = (
            (256, 128),
            (256, 256),
        )
        ln_triton_dim = 4
        return (
            self.training
            and x.dim() == ln_triton_dim
            and x.shape[-2:] in ln_triton_shapes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._should_use_triton_kernels(x):
            return LayerNormSmallShapeOptImpl.apply(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

# To load auto tuned cache:
from apex.contrib.openfold_triton._layer_norm_config_ampere import _auto_tuned_config_ampere
from apex.contrib.openfold_triton._layer_norm_config_hopper import _auto_tuned_config_hopper
from apex.contrib.openfold_triton import _tuneable_triton_kernels

def load_triton_auto_tuned_cache(dap_size: int, arch_type: str) -> None:
    auto_tuned_config = {
        "hopper": _auto_tuned_config_hopper,
        "ampere": _auto_tuned_config_ampere,
    }[arch_type]
    config_for_current_dap = auto_tuned_config[dap_size]
    for func_name, cache in config_for_current_dap.items():
        _tuneable_triton_kernels[func_name].cache = cache

load_triton_auto_tuned_cache(
    dap_size=4,  # supported values: 0, 1, 2, 4, 8
    arch_type="hopper",
)

```

## FusedAdamSWA

```python
from apex.contrib.openfold_triton.fused_adam_swa import FusedAdamSWA

fused_optimizer = FusedAdamSWA.from_optim(
    adam_optimizer=adam_optimizer,  # standard pytorch optimizer
    fp32_params=fp32_params,  # FP32 used in weight update
    bf16_params=bf16_params,  # BF16 used in forward, backward, reduction
    swa_params=swa_params,  # SWA used for evaluation
    swa_decay_rate=swa_decay_rate,  # for example: 0.9, 0.99, 0.999
)

fused_optimizer.step()  # fused optimizer step: casting BF16/FP32 + param updates + SWA

```
