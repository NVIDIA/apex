# © 2023 NVIDIA CORPORATION & AFFILIATES

from triton import Config

# Mapping schema: Dict[
#   dap_size: int, Dict[
#     function_name: str, Dict[
#       input_shape: Tuple[int, int], config: triton.Config
#     ]
#   ]
# ]
_auto_tuned_config_ampere = {
    0: {
        "_layer_norm_backward_dw_db_partial": {
            (65536, 128): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            ),
            (32768, 256): Config(
                {"N_BLOCK": 32, "M_PARTIAL_REDUCE": 256}, num_warps=8, num_stages=2
            ),
        },
        "_layer_norm_backward_dw_db_partial_strided": {
            (65536, 128): Config(
                {"N_BLOCK": 32, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            )
        },
        "_layer_norm_backward_dx": {
            (65536, 128): Config({"M_BLOCK": 4}, num_warps=2, num_stages=2),
            (32768, 256): Config({"M_BLOCK": 4}, num_warps=2, num_stages=2),
        },
        "_layer_norm_backward_dx_strided": {
            (65536, 128): Config({"M_BLOCK": 2}, num_warps=1, num_stages=2)
        },
        "_layer_norm_forward": {
            (65536, 128): Config({"M_BLOCK": 32}, num_warps=8, num_stages=2),
            (32768, 256): Config({"M_BLOCK": 16}, num_warps=8, num_stages=2),
        },
        "_layer_norm_forward_strided": {
            (65536, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2)
        },
    },
    2: {
        "_layer_norm_backward_dw_db_partial": {
            (65536, 128): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            ),
            (32768, 128): Config(
                {"N_BLOCK": 32, "M_PARTIAL_REDUCE": 256}, num_warps=8, num_stages=2
            ),
            (16384, 256): Config(
                {"N_BLOCK": 32, "M_PARTIAL_REDUCE": 256}, num_warps=8, num_stages=2
            ),
        },
        "_layer_norm_backward_dw_db_partial_strided": {
            (32768, 128): Config(
                {"N_BLOCK": 32, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            )
        },
        "_layer_norm_backward_dx": {
            (65536, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2),
            (32768, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2),
            (16384, 256): Config({"M_BLOCK": 4}, num_warps=2, num_stages=2),
        },
        "_layer_norm_backward_dx_strided": {
            (32768, 128): Config({"M_BLOCK": 2}, num_warps=1, num_stages=2)
        },
        "_layer_norm_forward": {
            (65536, 128): Config({"M_BLOCK": 32}, num_warps=8, num_stages=2),
            (32768, 128): Config({"M_BLOCK": 32}, num_warps=8, num_stages=2),
            (16384, 256): Config({"M_BLOCK": 16}, num_warps=8, num_stages=2),
        },
        "_layer_norm_forward_strided": {
            (32768, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2)
        },
    },
    4: {
        "_layer_norm_backward_dw_db_partial": {
            (65536, 128): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            ),
            (16384, 128): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 256}, num_warps=8, num_stages=2
            ),
            (8192, 256): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 256}, num_warps=8, num_stages=2
            ),
        },
        "_layer_norm_backward_dw_db_partial_strided": {
            (16384, 128): Config(
                {"N_BLOCK": 32, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            )
        },
        "_layer_norm_backward_dx": {
            (65536, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2),
            (16384, 128): Config({"M_BLOCK": 4}, num_warps=2, num_stages=2),
            (8192, 256): Config({"M_BLOCK": 1}, num_warps=1, num_stages=2),
        },
        "_layer_norm_backward_dx_strided": {
            (16384, 128): Config({"M_BLOCK": 2}, num_warps=1, num_stages=2)
        },
        "_layer_norm_forward": {
            (65536, 128): Config({"M_BLOCK": 32}, num_warps=8, num_stages=2),
            (16384, 128): Config({"M_BLOCK": 32}, num_warps=8, num_stages=2),
            (8192, 256): Config({"M_BLOCK": 16}, num_warps=8, num_stages=2),
        },
        "_layer_norm_forward_strided": {
            (16384, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2)
        },
    },
    8: {
        "_layer_norm_backward_dw_db_partial": {
            (65536, 128): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            ),
            (8192, 128): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 256}, num_warps=8, num_stages=2
            ),
            (4096, 256): Config(
                {"N_BLOCK": 16, "M_PARTIAL_REDUCE": 256}, num_warps=8, num_stages=2
            ),
        },
        "_layer_norm_backward_dw_db_partial_strided": {
            (8192, 128): Config(
                {"N_BLOCK": 32, "M_PARTIAL_REDUCE": 512}, num_warps=8, num_stages=2
            )
        },
        "_layer_norm_backward_dx": {
            (65536, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2),
            (8192, 128): Config({"M_BLOCK": 2}, num_warps=1, num_stages=2),
            (4096, 256): Config({"M_BLOCK": 1}, num_warps=1, num_stages=2),
        },
        "_layer_norm_backward_dx_strided": {
            (8192, 128): Config({"M_BLOCK": 1}, num_warps=1, num_stages=2)
        },
        "_layer_norm_forward": {
            (65536, 128): Config({"M_BLOCK": 32}, num_warps=8, num_stages=2),
            (8192, 128): Config({"M_BLOCK": 16}, num_warps=8, num_stages=2),
            (4096, 256): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2),
        },
        "_layer_norm_forward_strided": {
            (8192, 128): Config({"M_BLOCK": 8}, num_warps=4, num_stages=2)
        },
    },
}

_auto_tuned_config_ampere[1] = _auto_tuned_config_ampere[0]
