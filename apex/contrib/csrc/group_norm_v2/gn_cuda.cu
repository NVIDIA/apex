#include "gn.hpp"

#include <cstdio>
#include <mutex>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "gn_utils.hpp"
#include "gn_dispatch_hw_c.hpp"


#define DISPATCH_G_SILU(g, silu, G, SILU, ...) [&] { \
    if (g == 16 && silu == (g == 16)) { constexpr int G = 16; constexpr bool SILU = G == 16; return __VA_ARGS__(); } \
    if (g == 32 && silu == (g == 16)) { constexpr int G = 32; constexpr bool SILU = G == 16; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_G_SILU " + std::to_string(g) + " " + std::to_string(silu)); \
    }()

namespace group_norm_v2 {

template<typename T, int HW, int C, int G, bool SILU>
void gn_cuda_single_shape(GN_CUDA_HOST_PARAMS(T));

template<typename T, int HW, int C, int G, bool SILU>
void gn_bwd_cuda_single_shape(GN_BWD_CUDA_HOST_PARAMS(T));

template<typename T>
void gn_cuda(GN_CUDA_HOST_PARAMS(T)) {
    DISPATCH_HW_C(hw, num_groups * channels_per_group, HW, C, [&] {
        DISPATCH_G_SILU(num_groups, silu, G, SILU, [&] {
            return gn_cuda_single_shape<T, HW, C, G, SILU>(GN_CUDA_HOST_ARGS);
        });
    });
}

template<typename T>
void gn_bwd_cuda(GN_BWD_CUDA_HOST_PARAMS(T)) {
    DISPATCH_HW_C(hw, num_groups * channels_per_group, HW, C, [&] {
        DISPATCH_G_SILU(num_groups, silu, G, SILU, [&] {
            return gn_bwd_cuda_single_shape<T, HW, C, G, SILU>(GN_BWD_CUDA_HOST_ARGS);
        });
    });
}

template void gn_cuda(GN_CUDA_HOST_PARAMS(half));
template void gn_cuda(GN_CUDA_HOST_PARAMS(__nv_bfloat16));

template void gn_bwd_cuda(GN_BWD_CUDA_HOST_PARAMS(half));
template void gn_bwd_cuda(GN_BWD_CUDA_HOST_PARAMS(__nv_bfloat16));


}
