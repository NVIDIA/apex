#pragma once

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "gn_utils.hpp"
#include "gn_cuda_kernel.cuh"


namespace group_norm_v2 {

#define DISPATCH_LOWER_BOUND_N(VALUE, CONST_NAME, ...) [&] { \
    if (VALUE >= 16) { constexpr int CONST_NAME = 16; return __VA_ARGS__(); } \
    if (VALUE >= 8) { constexpr int CONST_NAME = 8; return __VA_ARGS__(); } \
    if (VALUE >= 4) { constexpr int CONST_NAME = 4; return __VA_ARGS__(); } \
    if (VALUE >= 2) { constexpr int CONST_NAME = 2; return __VA_ARGS__(); } \
    if (VALUE >= 1) { constexpr int CONST_NAME = 1; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_LOWER_BOUND_N " + std::to_string(VALUE)); \
    }()

#define DISPATCH_CUDA_ARCH_AND_LOWER_BOUND_SM_COUNT(runtime_cuda_arch, sm_count, RUNTIME_CUDA_ARCH, LB_SM_COUNT, ...) [&] { \
    if (runtime_cuda_arch == 1000 && sm_count >= 148 && sm_count == 148) { constexpr int RUNTIME_CUDA_ARCH = 1000, LB_SM_COUNT = 148; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_CUDA_ARCH_AND_LOWER_BOUND_SM_COUNT " + std::to_string(runtime_cuda_arch) + " " + std::to_string(sm_count)); \
    }()

#define DISPATCH_SM_MARGIN(VALUE, CONST_NAME, ...) [&] { \
    if (VALUE == 0) { constexpr int CONST_NAME = 0; return __VA_ARGS__(); } \
    if (VALUE == 32) { constexpr int CONST_NAME = 32; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_SM_MARGIN " + std::to_string(VALUE)); \
    }()


inline constexpr int get_max_cuda_arch() {
    int cuda_arch_list[] = {__CUDA_ARCH_LIST__};
    int max_cuda_arch = -1;
    for (int cuda_arch_item : cuda_arch_list) {
        if (cuda_arch_item > max_cuda_arch) {
            max_cuda_arch = cuda_arch_item;
        }
    }
    return max_cuda_arch;
}

template<typename T, bool BWD, bool REQUIRES_WGRAD, int HW, int G, int CPG, int LB_N, int RUNTIME_CUDA_ARCH, int LB_SM_COUNT, int EFFECTIVE_CUDA_ARCH, int SM_MARGIN>
constexpr auto compute_gn_params() {
    constexpr int C = G * CPG;

    int BLOCK_DIM_X;
    int C_PER_BLOCK;
    int ROWS_PER_BLOCK;
    bool LOAD_TWICE;
    int BLOCKS_PER_SM;
    WgradSyncMethod wgrad_sync_method = WGRAD_SYNC_UNSPECIFIED;

    if (HW * CPG * (1 + BWD) * sizeof(T) <= 20480) {
        // block sync
        C_PER_BLOCK = CPG;
        ROWS_PER_BLOCK = HW;
        BLOCK_DIM_X = lcm(32, C_PER_BLOCK);
        while (BLOCK_DIM_X < 256) {
            BLOCK_DIM_X *= 2;
        }
        BLOCKS_PER_SM = 1;
        LOAD_TWICE = BLOCKS_PER_SM * ROWS_PER_BLOCK * C_PER_BLOCK * (1 + BWD) * sizeof(T) > 36000 * 4;
    } else {
        // virtual cluster sync
        int c_per_cluster = lcm(128 / (int)sizeof(T), CPG);

        C_PER_BLOCK = c_per_cluster;
        BLOCK_DIM_X = C_PER_BLOCK == 320 ? 320 : 480;

        int one_pass_max_rows = 36000 * 4 / (C_PER_BLOCK * (1 + BWD) * sizeof(T));

        std::tuple<bool, int, int, int, int, int> best_candidate{};
        BLOCKS_PER_SM = 0;
        ROWS_PER_BLOCK = 0;
        for (int blocks_per_sm = 1; blocks_per_sm <= 3; blocks_per_sm++) {
            for (int rows_per_block = HW; rows_per_block >= 1; rows_per_block /= 2) {
                int virtual_cluster_size = (HW / rows_per_block) * (c_per_cluster / C_PER_BLOCK);
                if (virtual_cluster_size > blocks_per_sm * (LB_SM_COUNT - SM_MARGIN)) {
                    continue;
                }
                int num_clusters = blocks_per_sm * (LB_SM_COUNT - SM_MARGIN) / virtual_cluster_size;
                int num_tasks = LB_N * (C / c_per_cluster);
                int num_waves = up_div(num_tasks, num_clusters);
                bool load_twice = rows_per_block > one_pass_max_rows / blocks_per_sm;
                int wave_util = 10000 * std::min(num_tasks, num_clusters) * virtual_cluster_size / (blocks_per_sm * (LB_SM_COUNT - SM_MARGIN));
                decltype(best_candidate) candidate = {
                    true,
                    !load_twice,
                    !(num_waves >= 2 && blocks_per_sm == 1),  // overlap
                    -num_waves,
                    std::min(9000, wave_util),
                    -blocks_per_sm,
                };
                if (candidate > best_candidate) {
                    best_candidate = candidate;
                    BLOCKS_PER_SM = blocks_per_sm;
                    ROWS_PER_BLOCK = rows_per_block;
                }
            }
        }

        LOAD_TWICE = ROWS_PER_BLOCK > one_pass_max_rows / BLOCKS_PER_SM;
    }

    int c_per_cluster = lcm(CPG, C_PER_BLOCK);
    int virtual_cluster_size = (c_per_cluster / C_PER_BLOCK) * (HW / ROWS_PER_BLOCK);
    bool HARDWARE_CLUSTER = virtual_cluster_size <= 2 && virtual_cluster_size != 1 && SM_MARGIN == 0;
    int MAX_VEC_BYTES = 8;
    int VEC_ELEMS = std::min(gcd(MAX_VEC_BYTES / (int)sizeof(T), C_PER_BLOCK),
                             gcd(MAX_VEC_BYTES / (int)sizeof(T), ROWS_PER_BLOCK * C_PER_BLOCK / BLOCK_DIM_X));

    return std::make_tuple(
        BLOCK_DIM_X,
        C_PER_BLOCK,
        ROWS_PER_BLOCK,
        VEC_ELEMS,
        LOAD_TWICE,
        BLOCKS_PER_SM,
        HARDWARE_CLUSTER,
        wgrad_sync_method
    );
}

template<int EFFECTIVE_CUDA_ARCH>
class CompileCondition {
public:
    // Save compilation time for unused CUDA_ARCHs
    __host__ __device__ static constexpr bool matches() {
#if defined(__CUDA_ARCH__)
        return __CUDA_ARCH__ == EFFECTIVE_CUDA_ARCH;
#else
        return false;
#endif
    }
};

template<typename T, int HW, int C, int G, bool SILU>
void gn_cuda_single_shape(GN_CUDA_HOST_PARAMS(T)) {
    if (out == x) {
        throw std::invalid_argument("not __restrict__");
    }

    cudaDeviceProp const &deviceProp = get_device_prop(device_id);
    int runtime_cuda_arch = deviceProp.major * 100 + deviceProp.minor * 10;
    int sm_count = deviceProp.multiProcessorCount;

    DISPATCH_LOWER_BOUND_N(n, LB_N, [&] {
        DISPATCH_CUDA_ARCH_AND_LOWER_BOUND_SM_COUNT(runtime_cuda_arch, sm_count, RUNTIME_CUDA_ARCH, LB_SM_COUNT, [&] {
        DISPATCH_SM_MARGIN(sm_margin, SM_MARGIN, [&] {
            if (hw != HW) {
                throw std::invalid_argument("wrong HW");
            }
            if (num_groups * channels_per_group != C) {
                throw std::invalid_argument("wrong C");
            }
            if (num_groups != G) {
                throw std::invalid_argument("wrong G");
            }
            if (silu != SILU) {
                throw std::invalid_argument("wrong SILU");
            }
            if (n < LB_N) {
                throw std::invalid_argument("wrong LB_N");
            }
            if (runtime_cuda_arch != RUNTIME_CUDA_ARCH) {
                throw std::invalid_argument("wrong RUNTIME_CUDA_ARCH");
            }
            if (sm_count < LB_SM_COUNT) {
                throw std::invalid_argument("wrong LB_SM_COUNT");
            }
            if (sm_margin != SM_MARGIN) {
                throw std::invalid_argument("wrong SM_MARGIN");
            }
            constexpr int EFFECTIVE_CUDA_ARCH = std::min(RUNTIME_CUDA_ARCH, get_max_cuda_arch());

            constexpr int CPG = C / G;

            constexpr auto params = compute_gn_params<T, false, false, HW, G, CPG, LB_N, RUNTIME_CUDA_ARCH, LB_SM_COUNT, EFFECTIVE_CUDA_ARCH, SM_MARGIN>();
            constexpr int BLOCK_DIM_X =       std::get<0>(params);
            constexpr int C_PER_BLOCK =       std::get<1>(params);
            constexpr int ROWS_PER_BLOCK =    std::get<2>(params);
            constexpr int VEC_ELEMS =         std::get<3>(params);
            constexpr bool LOAD_TWICE =       std::get<4>(params);
            constexpr int BLOCKS_PER_SM =     std::get<5>(params);
            constexpr bool HARDWARE_CLUSTER = std::get<6>(params);

            constexpr int C_PER_CLUSTER = lcm(CPG, C_PER_BLOCK);
            constexpr int VIRTUAL_CLUSTER_SIZE = (C_PER_CLUSTER / C_PER_BLOCK) * (HW / ROWS_PER_BLOCK);
            constexpr int NUM_VIRTUAL_CLUSTERS = ((LB_SM_COUNT - SM_MARGIN) * BLOCKS_PER_SM) / VIRTUAL_CLUSTER_SIZE;
            constexpr bool PERSISTENT = !HARDWARE_CLUSTER && VIRTUAL_CLUSTER_SIZE >= 2;

            if (meta_ptr) {
                constexpr int MAX_NUM_GROUPS_PER_BLOCK = C_PER_BLOCK % CPG == 0 ? C_PER_BLOCK / CPG : up_div(C_PER_BLOCK - gcd(C_PER_BLOCK, CPG), CPG) + 1;
                meta_ptr->red_buffer_size = 2 * NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK * 2;
                meta_ptr->barrier_size = NUM_VIRTUAL_CLUSTERS;
                meta_ptr->BLOCK_DIM_X = BLOCK_DIM_X;
                meta_ptr->C_PER_BLOCK = C_PER_BLOCK;
                meta_ptr->ROWS_PER_BLOCK = ROWS_PER_BLOCK;
                meta_ptr->VEC_ELEMS = VEC_ELEMS;
                meta_ptr->LOAD_TWICE = LOAD_TWICE;
                meta_ptr->BLOCKS_PER_SM = BLOCKS_PER_SM;
                meta_ptr->HARDWARE_CLUSTER = HARDWARE_CLUSTER;
                meta_ptr->wgrad_sync_method = (int)WGRAD_SYNC_UNSPECIFIED;
            }
            if (meta_only) {
                return;
            }

            cudaLaunchConfig_t config = {0};
            // The grid dimension is not affected by cluster launch, and is still enumerated
            // using number of blocks.
            // The grid dimension should be a multiple of cluster size.
            config.gridDim = dim3(VIRTUAL_CLUSTER_SIZE, PERSISTENT ? std::min((int)n * (C / C_PER_CLUSTER), NUM_VIRTUAL_CLUSTERS) : n * (C / C_PER_CLUSTER), 1);
            config.blockDim = BLOCK_DIM_X;
            config.stream = stream;

            cudaLaunchAttribute attribute[2];
            if constexpr (HARDWARE_CLUSTER) {
                attribute[0].id = cudaLaunchAttributeClusterDimension;
                attribute[0].val.clusterDim.x = VIRTUAL_CLUSTER_SIZE; // Cluster size in X-dimension
                attribute[0].val.clusterDim.y = 1;
                attribute[0].val.clusterDim.z = 1;
                config.attrs = attribute;
                config.numAttrs++;
            }
            if constexpr (PERSISTENT) {
                attribute[config.numAttrs].id = cudaLaunchAttributeCooperative;
                attribute[config.numAttrs].val.cooperative = 1;
                config.attrs = attribute;
                config.numAttrs++;
            }

            auto kernel = &gn_cuda_kernel<T, BLOCK_DIM_X, BLOCKS_PER_SM, G, CPG, HW, SILU, ROWS_PER_BLOCK, C_PER_BLOCK, C_PER_CLUSTER, VEC_ELEMS, PERSISTENT, NUM_VIRTUAL_CLUSTERS, LOAD_TWICE, HARDWARE_CLUSTER, CompileCondition<EFFECTIVE_CUDA_ARCH> >;
            if constexpr (HARDWARE_CLUSTER) {
                if constexpr (VIRTUAL_CLUSTER_SIZE > 8) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
                }
                int max_cluster_size;
                int active_clusters;
                CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)kernel, &config));
                if (VIRTUAL_CLUSTER_SIZE <= max_cluster_size && PERSISTENT) {
                    attribute[0].val.clusterDim.x = VIRTUAL_CLUSTER_SIZE;
                    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&active_clusters, (void *)kernel, &config));
                }
                if (VIRTUAL_CLUSTER_SIZE <= max_cluster_size && (!PERSISTENT || PERSISTENT && NUM_VIRTUAL_CLUSTERS <= active_clusters)) {
                    attribute[0].val.clusterDim.x = VIRTUAL_CLUSTER_SIZE;
                } else {
                    constexpr bool HARDWARE_CLUSTER_NEW = false;
                    constexpr bool PERSISTENT_NEW = !HARDWARE_CLUSTER_NEW && VIRTUAL_CLUSTER_SIZE >= 2;
                    config.gridDim = dim3(VIRTUAL_CLUSTER_SIZE, PERSISTENT_NEW ? std::min((int)n * (C / C_PER_CLUSTER), NUM_VIRTUAL_CLUSTERS) : n * (C / C_PER_CLUSTER), 1);
                    config.attrs = nullptr;
                    config.numAttrs = 0;
                    if constexpr (PERSISTENT_NEW) {
                        attribute[config.numAttrs].id = cudaLaunchAttributeCooperative;
                        attribute[config.numAttrs].val.cooperative = 1;
                        config.attrs = attribute;
                        config.numAttrs++;
                    }
                    kernel = &gn_cuda_kernel<T, BLOCK_DIM_X, BLOCKS_PER_SM, G, CPG, HW, SILU, ROWS_PER_BLOCK, C_PER_BLOCK, C_PER_CLUSTER, VEC_ELEMS, PERSISTENT_NEW, NUM_VIRTUAL_CLUSTERS, LOAD_TWICE, HARDWARE_CLUSTER_NEW, CompileCondition<EFFECTIVE_CUDA_ARCH> >;
                }
            }
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, w, b, eps, n, mean_var_out, red_buffer, barrier));
        });
        });
    });
}


template<typename T, int HW, int C, int G, bool SILU>
void gn_bwd_cuda_single_shape(GN_BWD_CUDA_HOST_PARAMS(T)) {
    if (grad_input == grad_output || grad_input == x) {
        throw std::invalid_argument("not __restrict__");
    }

    cudaDeviceProp const &deviceProp = get_device_prop(device_id);
    int runtime_cuda_arch = deviceProp.major * 100 + deviceProp.minor * 10;
    int sm_count = deviceProp.multiProcessorCount;

    DISPATCH_LOWER_BOUND_N(n, LB_N, [&] {
        DISPATCH_CUDA_ARCH_AND_LOWER_BOUND_SM_COUNT(runtime_cuda_arch, sm_count, RUNTIME_CUDA_ARCH, LB_SM_COUNT, [&] {
        DISPATCH_SM_MARGIN(sm_margin, SM_MARGIN, [&] {
            if (hw != HW) {
                throw std::invalid_argument("wrong HW");
            }
            if (num_groups * channels_per_group != C) {
                throw std::invalid_argument("wrong C");
            }
            if (num_groups != G) {
                throw std::invalid_argument("wrong G");
            }
            if (silu != SILU) {
                throw std::invalid_argument("wrong SILU");
            }
            if (n < LB_N) {
                throw std::invalid_argument("wrong LB_N");
            }
            if (runtime_cuda_arch != RUNTIME_CUDA_ARCH) {
                throw std::invalid_argument("wrong RUNTIME_CUDA_ARCH");
            }
            if (sm_count < LB_SM_COUNT) {
                throw std::invalid_argument("wrong LB_SM_COUNT");
            }
            if (sm_margin != SM_MARGIN) {
                throw std::invalid_argument("wrong SM_MARGIN");
            }
            constexpr int EFFECTIVE_CUDA_ARCH = std::min(RUNTIME_CUDA_ARCH, get_max_cuda_arch());

            constexpr bool REQUIRES_WGRAD = true;
            constexpr int CPG = C / G;

            constexpr auto params = compute_gn_params<T, true, REQUIRES_WGRAD, HW, G, CPG, LB_N, RUNTIME_CUDA_ARCH, LB_SM_COUNT, EFFECTIVE_CUDA_ARCH, SM_MARGIN>();
            constexpr int BLOCK_DIM_X =       std::get<0>(params);
            constexpr int C_PER_BLOCK =       std::get<1>(params);
            constexpr int ROWS_PER_BLOCK =    std::get<2>(params);
            constexpr int VEC_ELEMS =         std::get<3>(params);
            constexpr bool LOAD_TWICE =       std::get<4>(params);
            constexpr int BLOCKS_PER_SM =     std::get<5>(params);
            constexpr bool HARDWARE_CLUSTER = std::get<6>(params);
            constexpr WgradSyncMethod wgrad_sync_method_hint = std::get<7>(params);

            constexpr int C_PER_CLUSTER = lcm(CPG, C_PER_BLOCK);
            constexpr int VIRTUAL_CLUSTER_SIZE = (C_PER_CLUSTER / C_PER_BLOCK) * (HW / ROWS_PER_BLOCK);
            constexpr int NUM_VIRTUAL_CLUSTERS_NOT_ALIGNED = ((LB_SM_COUNT - SM_MARGIN) * BLOCKS_PER_SM) / VIRTUAL_CLUSTER_SIZE;

            // TODO: specilize for the case that REQUIRES_WGRAD == false
            constexpr bool PERSISTENT = true;
            constexpr WgradSyncMethod wgrad_sync_method =
                wgrad_sync_method_hint == WGRAD_SYNC_UNSPECIFIED ?
                    NUM_VIRTUAL_CLUSTERS_NOT_ALIGNED > 2 * (C / C_PER_CLUSTER) || NUM_VIRTUAL_CLUSTERS_NOT_ALIGNED % (C / C_PER_CLUSTER) == 0 ?
                        WGRAD_REUSE_SUM_SYNC_GROUP :
                        WGRAD_REUSE_SUM_SYNC_GRID :
                    wgrad_sync_method_hint;
            constexpr int NUM_VIRTUAL_CLUSTERS =
                wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GROUP || wgrad_sync_method == WGRAD_REUSE_SUM_SYNC_GROUP ?
                    NUM_VIRTUAL_CLUSTERS_NOT_ALIGNED / (C / C_PER_CLUSTER) * (C / C_PER_CLUSTER) :
                    NUM_VIRTUAL_CLUSTERS_NOT_ALIGNED;

            if (meta_ptr) {
                constexpr int MAX_NUM_GROUPS_PER_BLOCK = C_PER_BLOCK % CPG == 0 ? C_PER_BLOCK / CPG : up_div(C_PER_BLOCK - gcd(C_PER_BLOCK, CPG), CPG) + 1;
                meta_ptr->red_buffer_size = 2 * NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK * 2 +
                    std::max(n, (int64_t)NUM_VIRTUAL_CLUSTERS / (C / C_PER_CLUSTER)) * (HW / ROWS_PER_BLOCK) * C * 2;
                meta_ptr->barrier_size = NUM_VIRTUAL_CLUSTERS + C / C_PER_CLUSTER;
                meta_ptr->BLOCK_DIM_X = BLOCK_DIM_X;
                meta_ptr->C_PER_BLOCK = C_PER_BLOCK;
                meta_ptr->ROWS_PER_BLOCK = ROWS_PER_BLOCK;
                meta_ptr->VEC_ELEMS = VEC_ELEMS;
                meta_ptr->LOAD_TWICE = LOAD_TWICE;
                meta_ptr->BLOCKS_PER_SM = BLOCKS_PER_SM;
                meta_ptr->HARDWARE_CLUSTER = HARDWARE_CLUSTER;
                meta_ptr->wgrad_sync_method = (int)wgrad_sync_method;
            }
            if (meta_only) {
                return;
            }

            cudaLaunchConfig_t config = {0};
            // The grid dimension is not affected by cluster launch, and is still enumerated
            // using number of blocks.
            // The grid dimension should be a multiple of cluster size.
            config.gridDim = dim3(VIRTUAL_CLUSTER_SIZE, PERSISTENT ? NUM_VIRTUAL_CLUSTERS : n * (C / C_PER_CLUSTER), 1);
            config.blockDim = BLOCK_DIM_X;
            config.stream = stream;

            cudaLaunchAttribute attribute[2];
            if constexpr (HARDWARE_CLUSTER) {
                attribute[0].id = cudaLaunchAttributeClusterDimension;
                attribute[0].val.clusterDim.x = 1; // Cluster size in X-dimension
                attribute[0].val.clusterDim.y = 1;
                attribute[0].val.clusterDim.z = 1;
                config.attrs = attribute;
                config.numAttrs++;
            }
            if constexpr (PERSISTENT) {
                attribute[config.numAttrs].id = cudaLaunchAttributeCooperative;
                attribute[config.numAttrs].val.cooperative = 1;
                config.attrs = attribute;
                config.numAttrs++;
            }

            auto kernel = &gn_bwd_cuda_kernel<T, BLOCK_DIM_X, BLOCKS_PER_SM, G, CPG, HW, SILU, REQUIRES_WGRAD, ROWS_PER_BLOCK, C_PER_BLOCK, C_PER_CLUSTER, VEC_ELEMS, PERSISTENT, NUM_VIRTUAL_CLUSTERS, LOAD_TWICE, HARDWARE_CLUSTER, wgrad_sync_method, CompileCondition<EFFECTIVE_CUDA_ARCH> >;
            if constexpr (HARDWARE_CLUSTER) {
                if constexpr (VIRTUAL_CLUSTER_SIZE > 8) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
                }
                int max_cluster_size;
                int active_clusters;
                CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)kernel, &config));
                if (VIRTUAL_CLUSTER_SIZE <= max_cluster_size && PERSISTENT) {
                    attribute[0].val.clusterDim.x = VIRTUAL_CLUSTER_SIZE;
                    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&active_clusters, (void *)kernel, &config));
                }
                if (VIRTUAL_CLUSTER_SIZE <= max_cluster_size && (!PERSISTENT || PERSISTENT && NUM_VIRTUAL_CLUSTERS <= active_clusters)) {
                    attribute[0].val.clusterDim.x = VIRTUAL_CLUSTER_SIZE;
                } else {
                    attribute[0].val.clusterDim.x = 1;
                    kernel = &gn_bwd_cuda_kernel<T, BLOCK_DIM_X, BLOCKS_PER_SM, G, CPG, HW, SILU, REQUIRES_WGRAD, ROWS_PER_BLOCK, C_PER_BLOCK, C_PER_CLUSTER, VEC_ELEMS, PERSISTENT, NUM_VIRTUAL_CLUSTERS, LOAD_TWICE, false, wgrad_sync_method, CompileCondition<EFFECTIVE_CUDA_ARCH> >;
                }
            }
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, grad_input, grad_weight, grad_bias, grad_output, x, w, b, mean_var, eps, n, red_buffer, barrier));
        });
        });
    });
}

#define GN_CUDA_INST_DEFINE(HW, C) \
    template void gn_cuda_single_shape<half, HW, C, 16, true>(GN_CUDA_HOST_PARAMS(half)); \
    template void gn_cuda_single_shape<half, HW, C, 32, false>(GN_CUDA_HOST_PARAMS(half)); \
    template void gn_bwd_cuda_single_shape<half, HW, C, 16, true>(GN_BWD_CUDA_HOST_PARAMS(half)); \
    template void gn_bwd_cuda_single_shape<half, HW, C, 32, false>(GN_BWD_CUDA_HOST_PARAMS(half)); \
    template void gn_cuda_single_shape<__nv_bfloat16, HW, C, 16, true>(GN_CUDA_HOST_PARAMS(__nv_bfloat16)); \
    template void gn_cuda_single_shape<__nv_bfloat16, HW, C, 32, false>(GN_CUDA_HOST_PARAMS(__nv_bfloat16)); \
    template void gn_bwd_cuda_single_shape<__nv_bfloat16, HW, C, 16, true>(GN_BWD_CUDA_HOST_PARAMS(__nv_bfloat16)); \
    template void gn_bwd_cuda_single_shape<__nv_bfloat16, HW, C, 32, false>(GN_BWD_CUDA_HOST_PARAMS(__nv_bfloat16));

}
