/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file nhwc_batch_norm_kernel.h
 * \brief CUDA NHWC Batch Normalization code
 * \author Shankara Rao Thejaswi Nanditale, Dick Carter, Maxim Milakov, Evgeni Krimer
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_KERNEL_H_
#define MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_KERNEL_H_

#include <stdint.h>
#include <algorithm>

#define DEVICE_FUNCTION static inline __device__

// CTA margin used by cooperative launch. Can be overridden by env var NHWC_BATCHNORM_LAUNCH_MARGIN.
#define NHWC_BATCHNORM_LAUNCH_MARGIN_MIN     3
#define NHWC_BATCHNORM_LAUNCH_MARGIN_DEFAULT NHWC_BATCHNORM_LAUNCH_MARGIN_MIN

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int ELEMENTS_PER_LDG >
struct PackedStorage {
    enum { PACKED_ELEMENTS_PER_LDG = ELEMENTS_PER_LDG };
    typedef T Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int ELEMENTS_PER_LDG >
struct PackedStorage<uint16_t, ELEMENTS_PER_LDG> {
    enum { PACKED_ELEMENTS_PER_LDG = ELEMENTS_PER_LDG/2 };
    typedef int Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void from_float(int (&dst)[N], const float (&src)[2*N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        uint16_t lo, hi;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(lo) : "f"(src[2*i+0]));
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(hi) : "f"(src[2*i+1]));
        asm volatile("mov.b32 %0, {%1, %2};"  : "=r"(dst[i]) : "h"(lo), "h"(hi));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void from_float(float (&dst)[N], const float (&src)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void to_float(float (&dst)[2*N], int (&src)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        uint16_t lo, hi;
        asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(src[i]));
        asm volatile("cvt.f32.f16 %0, %1;"   : "=f"(dst[2*i+0])   : "h"(lo));
        asm volatile("cvt.f32.f16 %0, %1;"   : "=f"(dst[2*i+1])   : "h"(hi));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void to_float(float (&dst)[N], float (&src)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void ldg(int (&dst)[1], const uint16_t *gmem) {
    dst[0] = __ldg((const int*) gmem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void ldg_stream(int (&dst)[1], const uint16_t *gmem) {
    unsigned int tmp;
    asm volatile ("ld.global.cs.nc.s32 %0, [%1];"  : "=r"(tmp) : "l" ((const uint *)gmem));
    dst[0] = tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void ldg(int (&dst)[2], const uint16_t *gmem) {
    int2 tmp = __ldg((const int2*) gmem);
    dst[0] = tmp.x;
    dst[1] = tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void ldg_stream(int (&dst)[2], const uint16_t *gmem) {
    int2 tmp;
    asm volatile ("ld.global.cs.nc.v2.s32 {%0,%1}, [%2];"
        : "=r"(tmp.x), "=r"(tmp.y) : "l"((const int2 *)gmem));
    dst[0] = tmp.x;
    dst[1] = tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void ldg(float (&dst)[N], const uint16_t *gmem) {
    int tmp[N/2];
    ldg(tmp, gmem);
    to_float(dst, tmp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void ldg_stream(float (&dst)[N], const uint16_t *gmem) {
    int tmp[N/2];
    ldg_stream(tmp, gmem);
    to_float(dst, tmp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void stg(uint16_t *gmem, int (&src)[1]) {
    reinterpret_cast<int*>(gmem)[0] = src[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void stg_stream(uint16_t *gmem, int (&src)[1]) {
    unsigned int tmp = src[0];
    asm volatile ("st.global.cs.s32 [%0], %1;"
        :: "l"((uint *)gmem) , "r"(tmp));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void stg(uint16_t *gmem, int (&src)[2]) {
    reinterpret_cast<int2*>(gmem)[0] = make_int2(src[0], src[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void stg_stream(uint16_t *gmem, int (&src)[2]) {
    asm volatile ("st.global.cs.v2.s32 [%0], {%1,%2};"
        :: "l"((uint *)gmem) , "r"(src[0]), "r"( src[1]));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void stg(uint16_t *gmem, float (&src)[N]) {
    int tmp[N/2];
    from_float(tmp, src);
    stg(gmem, tmp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void stg_stream(uint16_t *gmem, float (&src)[N]) {
    int tmp[N/2];
    from_float(tmp, src);
    stg_stream(gmem, tmp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void read_from_gmem(float (&dst)[2], const float *gmem, int idx) {
    float2 tmp = __ldg(reinterpret_cast<const float2*>(&gmem[2*idx]));
    dst[0] = tmp.x;
    dst[1] = tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void read_from_gmem(float (&dst)[4], const float *gmem, int idx) {
    float4 tmp = __ldg(reinterpret_cast<const float4*>(&gmem[4*idx]));
    dst[0] = tmp.x;
    dst[1] = tmp.y;
    dst[2] = tmp.z;
    dst[3] = tmp.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void read_from_smem(float (&x)[2], const float *smem, int idx) {
    float2 tmp = *(const float2*) &smem[2*idx];
    x[0] = tmp.x;
    x[1] = tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void read_from_smem(int (&x)[1], const int *smem, int idx) {
    x[0] = smem[idx];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void read_from_smem(float (&x)[4], const float *smem, int idx) {
    float4 tmp = *(const float4*) &smem[4*idx];
    x[0] = tmp.x;
    x[1] = tmp.y;
    x[2] = tmp.z;
    x[3] = tmp.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void read_from_smem(int (&x)[2], const int *smem, int idx) {
    int2 tmp = *(const int2*) &smem[2*idx];
    x[0] = tmp.x;
    x[1] = tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void write_to_gmem(float *gmem, int idx, const float (&src)[2]) {
    reinterpret_cast<float2*>(&gmem[2*idx])[0] = make_float2(src[0], src[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void write_to_gmem(float *gmem, int idx, const float (&src)[4]) {
    reinterpret_cast<float4*>(&gmem[4*idx])[0] = make_float4(src[0], src[1], src[2], src[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void scaled_write_to_gmem(float *gmem, int idx, const float (&src)[4], const float coeff) {
    reinterpret_cast<float4*>(&gmem[4*idx])[0] = make_float4(src[0]*coeff, src[1]*coeff, src[2]*coeff, src[3]*coeff);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void write_to_smem(float *smem, int idx, const float (&x)[2]) {
    reinterpret_cast<float2*>(&smem[2*idx])[0] = make_float2(x[0], x[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void write_to_smem(int *smem, int idx, const int (&x)[1]) {
    smem[idx] = x[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void write_to_smem(float *smem, int idx, const float (&x)[4]) {
    reinterpret_cast<float4*>(&smem[4*idx])[0] = make_float4(x[0], x[1], x[2], x[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_FUNCTION void write_to_smem(int *smem, int idx, const int (&x)[2]) {
    reinterpret_cast<int2*>(&smem[2*idx])[0] = make_int2(x[0], x[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void zero_array(int (&dst)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
DEVICE_FUNCTION void zero_array(float (&dst)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = 0.f;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void add(float (&x)[N], const float (&y)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        x[i] += y[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void multiply(float (&x)[N], const float (&y)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        x[i] *= y[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void scale_(float (&x)[N], float scalar) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        x[i] *= scalar;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void normalize(float (&x)[N], const float (&bias)[N],
                               const float (&scale)[N], const float (&m1)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        x[i] = bias[i] + scale[i] * (x[i] - m1[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Storage>
DEVICE_FUNCTION Storage relu(Storage in) {
    Storage zero = (Storage)0.f;
    return (in < zero)? zero : in;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void relu_activation(float (&x)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        x[i] = relu(x[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template< int THREADS_PER_CTA >
DEVICE_FUNCTION void parallel_sums_16x2(float *smem, float (&x)[4], int nhw,
                                        void* params_my_data, void** params_pair_datas, int off,
                                        const int magic,
                                        const int sync_iters) {
    // The size of a warp.
    const int THREADS_PER_WARP = 32;
    // The number of warps in a CTA.
    const int WARPS_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;
    // The number of threads per pixel.
    const int THREADS_PER_PIXEL = 16;
    // The number of elements per ldg.
    const int ELEMENTS_PER_LDG = 4;
    // The number of reducing ops, each uses its own space : mean, var, dscale, dbias
    const int REDUCE_OPS = 4;
    // Maximum block.y supported - limited due to buffer allocation
    const int MAX_BLOCK_Y = 256;
    const int MAX_OFFSET = REDUCE_OPS*MAX_BLOCK_Y;
    // The warp decomposition.
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    // total size of data per sync iter
    const int data_total = MAX_OFFSET*THREADS_PER_PIXEL*ELEMENTS_PER_LDG*2;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
        x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL+lane_id);
    }

    // The warp leaders, write to SMEM.
    if (lane_id < THREADS_PER_PIXEL) {
        write_to_smem(smem, warp_id*THREADS_PER_PIXEL + lane_id, x);
    }

    // The data is in SMEM. Do the final reduction.
    __syncthreads();

    // The 1st warp does all the work.
    // We do the final reduction each half-warp sequentially reduces the final values.
    if (warp_id == 0) {
        read_from_smem(x, smem, threadIdx.x);

        #pragma unroll
        for (int offset = 1;
             offset < WARPS_PER_CTA/(THREADS_PER_WARP / THREADS_PER_PIXEL); ++offset) {
            float y[ELEMENTS_PER_LDG];
            // Read the mean and variance from the other pixel.
            read_from_smem(y, smem, threadIdx.x + offset*THREADS_PER_WARP);
            // Compute the updated sum.
            add(x, y);
        }

        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL+lane_id);
        }

        // Make sure the data was read from SMEM.
        __syncwarp();

        // Store the final values.
        if (threadIdx.x < THREADS_PER_PIXEL) {
        // probably could do it earlier, before sync

        for (int sync_iter=0; sync_iter < sync_iters; ++sync_iter) {
            //float* params_pair_data = (reinterpret_cast<float**>(params_pair_datas))[sync_iter];
            void* params_pair_data = params_pair_datas[sync_iter];

            // skip the space consumed by previous sync iterations
            const int xbuf_offset = sync_iter*data_total;
            // data starts after flags, but have to skip previous
            const int data_offset = xbuf_offset
                                    + off*ELEMENTS_PER_LDG*THREADS_PER_PIXEL*2
                                    + ELEMENTS_PER_LDG*threadIdx.x*2;

            // after sums for this GPU were computed, let CTA0 broadcast the sum to over GPU
            if (blockIdx.x == 0) {
                volatile float * write_data =
                    &((reinterpret_cast<float*>(params_pair_data))[data_offset]);

                // write the data to memory region to be reflected to other GPU
                asm volatile ("st.global.wt.v4.b32 [%0], {%1,%2,%3,%4};"
                    :: "l"(write_data) , "f"(x[0]), "r"(magic), "f"(x[2]), "r"(magic));

                asm volatile ("st.global.wt.v4.b32 [%0], {%1,%2,%3,%4};"
                    :: "l"(write_data+4) , "f"(x[1]), "r"(magic), "f"(x[3]), "r"(magic));
            }

            // now each CTA (on each GPU) reads the data written by CTA 0 of the other GPU
            volatile float * read_data =
                &((reinterpret_cast<float*>(params_my_data))[data_offset]);

            float other[4];
            uint32_t other_flag_a, other_flag_b;
            do {
                asm volatile ("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(other[0]), "=r"(other_flag_a), "=f"(other[2]), "=r"(other_flag_b) : "l"(read_data));
            } while ((other_flag_a != magic) || (other_flag_b != magic));

            do {
                asm volatile ("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(other[1]), "=r"(other_flag_a), "=f"(other[3]), "=r"(other_flag_b) : "l"(read_data+4));
            } while ((other_flag_a != magic) || (other_flag_b != magic));

            add(x, other);
        }
        // finally, after syncing up and accounting for partial sums from
        // other GPUs as required, write the result


            write_to_smem(smem, threadIdx.x, x);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int THREADS_PER_CTA >
DEVICE_FUNCTION void parallel_sums_8x4(float *smem, float (&x)[4], int nhw) {
    // The size of a warp.
    const int THREADS_PER_WARP = 32;
    // The number of warps in a CTA.
    const int WARPS_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;
    // The number of threads per pixel.
    const int THREADS_PER_PIXEL = 8;
    // The number of elements per ldg.
    const int ELEMENTS_PER_LDG = 4;
    // The warp decomposition.
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
        x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL+lane_id);
        x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL*2+lane_id);
    }

    // The warp leaders, write to SMEM.
    if (lane_id < THREADS_PER_PIXEL) {
        write_to_smem(smem, warp_id*THREADS_PER_PIXEL + lane_id, x);
    }

    // The data is in SMEM. Do the final reduction.
    __syncthreads();

    // The 1st warp does all the work.
    // We do the final reduction each half-warp sequentially reduces the final values.
    if (warp_id == 0) {
        read_from_smem(x, smem, threadIdx.x);

        #pragma unroll
        for (int offset = 1;
             offset < WARPS_PER_CTA/(THREADS_PER_WARP / THREADS_PER_PIXEL); ++offset) {
            float y[ELEMENTS_PER_LDG];
            // Read the mean and variance from the other pixel.
            read_from_smem(y, smem, threadIdx.x + offset*THREADS_PER_WARP);
            // Compute the updated sum.
            add(x, y);
        }

        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL+lane_id);
            x[i] += __shfl_sync(0xffffffffU, x[i], THREADS_PER_PIXEL*2+lane_id);
        }

        // Make sure the data was read from SMEM.
        __syncwarp();

        // Store the final values.
        if (threadIdx.x < THREADS_PER_PIXEL) {
            write_to_smem(smem, threadIdx.x, x);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int THREADS_PER_CTA, int THREADS_PER_PIXEL, int ELEMENTS_PER_LDG >
DEVICE_FUNCTION void parallel_sums(float *smem, float (&x)[ELEMENTS_PER_LDG], int nhw) {
    // The size of a warp.
    const int THREADS_PER_WARP = 32;
    // The number of warps in a CTA.
    const int WARPS_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;
    // The number of pixels computed by a single warp.
    const int PIXELS_PER_WARP = THREADS_PER_WARP / THREADS_PER_PIXEL;

    // The position in the warp.
    const int nhw_in_warp = nhw % PIXELS_PER_WARP;
    // The C in the warp.
    const int c_in_warp = threadIdx.x % THREADS_PER_PIXEL;

    // Store the values to shared memory.
    write_to_smem(smem, threadIdx.x, x);

    // Compute the parallel sums.
    for (int offset = PIXELS_PER_WARP/2; offset > 0; offset /= 2) {
        // NOP.
        __syncwarp();

        // Read the running sum from the other thread.
        float y[ELEMENTS_PER_LDG];
        if (nhw_in_warp < offset) {
            read_from_smem(y, smem, threadIdx.x + offset*THREADS_PER_PIXEL);
        }

        // Compute the updated sum.
        add(x, y);

        // NOP.
        __syncwarp();

        // Update the sum in SMEM.
        if (offset > 1 && nhw_in_warp < offset) {
            write_to_smem(smem, threadIdx.x, x);
        }
    }

    // The warps are done. Do the final reduction at the CTA level.
    __syncthreads();

    // The warp leaders, write to SMEM.
    const int idx = (threadIdx.x/THREADS_PER_WARP)*THREADS_PER_PIXEL + c_in_warp;
    if (nhw_in_warp == 0) {
        write_to_smem(smem, idx, x);
    }

    // The data is in SMEM. Do the final reduction.
    __syncthreads();

    // Read the 1st element to prepare the work.
    if (nhw < WARPS_PER_CTA/2) {
        read_from_smem(x, smem, threadIdx.x);
    }

    // We have the running mean and running m2. Let's build the mean/var of the CTA.
    for (int offset = WARPS_PER_CTA/2; offset > 0; offset /= 2) {
        // NOP.
        __syncwarp();

        // Read the mean and variance from the other pixel.
        float y[ELEMENTS_PER_LDG];
        if (nhw < offset) {
            read_from_smem(y, smem, threadIdx.x + offset*THREADS_PER_PIXEL);
        }

        // Compute the updated sum.
        add(x, y);

        // NOP.
        __syncwarp();

        // Store the mean/var for the different pixels.
        if (nhw < offset) {
            write_to_smem(smem, threadIdx.x, x);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int THREADS_PER_PIXEL, int ELEMENTS_PER_LDG >
struct ParallelSums {
    template< int THREADS_PER_CTA >
    DEVICE_FUNCTION void dispatch(float *smem, float (&x)[ELEMENTS_PER_LDG], int nhw) {
        parallel_sums<THREADS_PER_CTA, THREADS_PER_PIXEL, ELEMENTS_PER_LDG>(smem, x, nhw);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct ParallelSums<16, 4> {
    template< int THREADS_PER_CTA >
    DEVICE_FUNCTION void dispatch(float *smem, float (&x)[4], int nhw) {
        parallel_sums_16x2<THREADS_PER_CTA>(smem, x, nhw, 0, 0, 0, 0, 0);
    }

    template< int THREADS_PER_CTA >
    DEVICE_FUNCTION void dispatchX(float *smem, float (&x)[4], int nhw, void* params_my_data, void** params_pair_datas, int off, const int magic, const unsigned int& sync_iters) {
        parallel_sums_16x2<THREADS_PER_CTA>(smem, x, nhw, params_my_data, params_pair_datas, off, magic, sync_iters);
    }
};

template<>
struct ParallelSums<8, 4> {
    template< int THREADS_PER_CTA >
    DEVICE_FUNCTION void dispatch(float *smem, float (&x)[4], int nhw) {
        parallel_sums_8x4<THREADS_PER_CTA>(smem, x, nhw);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int div_up(int m, int n) {
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// It is expected that all threads in the CTA enter this function!
DEVICE_FUNCTION void inter_block_sync(int* gmem_retired_ctas, int expected_count, bool master) {

    // Register the CTA.
    if (threadIdx.x == 0) {
        // Issue the membar.
        __threadfence();
        // Notify that the CTA is done.
        int val_to_add = 1;
        if (master) {
            val_to_add = -(expected_count - 1);
        }
        atomicAdd(gmem_retired_ctas, val_to_add);
    }

    // Are all CTAs done?
    if (threadIdx.x == 0) {
        int retired_ctas = -1;
        do {
            __threadfence();
            asm volatile ("ld.global.cg.b32 %0, [%1];"
                : "=r"(retired_ctas) : "l"(gmem_retired_ctas));
        } while (retired_ctas != 0);
    }
    __syncthreads();

}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct NhwcBatchNormFwdInferenceParams {
    // The input/output tensors.
    uint16_t *gmem_src, *gmem_dst, *gmem_src1;
    // the final mean and variance as calculated during the training process
    float *gmem_mean, *gmem_var;
    // The bias/scale.
    float *gmem_bias, *gmem_scale;
    // The dimensions.
    int nhw, c;
    // epsilon
    float var_eps;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// No DESIRED_OCCUPANCY launch bounds needed, as this is not launched cooperatively
template<
    typename Storage,
    int THREADS_PER_CTA,
    int THREADS_PER_PIXEL,
    int ELEMENTS_PER_LDG,
    bool USE_RELU,
    bool USE_ADD_RELU
>
__global__ __launch_bounds__(THREADS_PER_CTA)
    void nhwc_batch_norm_fwd_inference(NhwcBatchNormFwdInferenceParams params) {
    // The number of pixels loaded in a single LDG.
    const int PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
    // The number of C elements per CTA.
    const int C_ELEMENTS_PER_CTA = THREADS_PER_PIXEL*ELEMENTS_PER_LDG;

    // The start position in the NHW dimension where the CTA starts.
    const int cta_nhw_stride = gridDim.x * PIXELS_PER_LDG;
    // Compute the NHW coordinate of the thread in the CTA.
    const int thread_in_cta_nhw = threadIdx.x / THREADS_PER_PIXEL;
    // thread's starting point in NHW
    const int thread_nhw = thread_in_cta_nhw + blockIdx.x * PIXELS_PER_LDG;

    // The position in the C dimension where the CTA starts.
    const int cta_c = blockIdx.y * C_ELEMENTS_PER_CTA;
    // Compute the C coordinate of the thread in the CTA.
    const int thread_in_cta_c = threadIdx.x % THREADS_PER_PIXEL;
    // Compute the C coordinate of the thread.
    const int thread_c = cta_c + thread_in_cta_c*ELEMENTS_PER_LDG;

    // Is the thread working on a valid C dimension?
    const int is_valid_c = thread_c < params.c;

    float mean[ELEMENTS_PER_LDG], var[ELEMENTS_PER_LDG];
    float scale[ELEMENTS_PER_LDG], bias[ELEMENTS_PER_LDG];
    zero_array(mean);
    zero_array(var);
    zero_array(scale);
    zero_array(bias);
    if (is_valid_c) {
        read_from_gmem(var, &params.gmem_var[cta_c], thread_in_cta_c);
        read_from_gmem(scale, &params.gmem_scale[cta_c], thread_in_cta_c);
        read_from_gmem(mean, &params.gmem_mean[cta_c], thread_in_cta_c);
        read_from_gmem(bias, &params.gmem_bias[cta_c], thread_in_cta_c);
    }

    // Update the scale with the stddev and eps.
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
        scale[i] *= rsqrtf(var[i] + params.var_eps);
    }

    // The base pointers for reading/writing
    uint16_t *const gmem_src = &params.gmem_src[thread_c];
    uint16_t *const gmem_dst = &params.gmem_dst[thread_c];
    const uint16_t *gmem_src1 = nullptr;
    if (USE_ADD_RELU) {
        gmem_src1 = &params.gmem_src1[thread_c];
    }

    // apply BN
    for (int nhw = thread_nhw; nhw < params.nhw; nhw += cta_nhw_stride) {
        float x_math[ELEMENTS_PER_LDG];
        zero_array(x_math);
        if (is_valid_c) {
            ldg(x_math, &gmem_src[nhw*params.c]);
        }

        // Normalize and apply activation function
        normalize(x_math, bias, scale, mean);
        if (USE_ADD_RELU) {
            float x1_math[ELEMENTS_PER_LDG];
            ldg(x1_math, &gmem_src1[nhw*params.c]);
            add(x_math, x1_math);
            relu_activation(x_math);
        } else if (USE_RELU) {
            relu_activation(x_math);
        }

        if (is_valid_c) {
            stg(&gmem_dst[nhw*params.c], x_math);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct NhwcBatchNormFwdParams {
    // The input/output tensors.
    uint16_t *gmem_src, *gmem_dst, *gmem_src1;
    // The bias/scale.
    float *gmem_bias, *gmem_scale;
    // running mean/var (refer BN API from cudnn doc)
    float *gmem_running_mean, *gmem_running_var;
    // saved mean/var (refer BN API from cudnn doc)
    float *gmem_saved_mean, *gmem_saved_var;
    // ReLU bitmask
    unsigned int *gmem_relu_bitmask;
    // The dimensions.
    int nhw, c;
    // factor to scale sum of squared errors to get saved variance.  Must be 1/nhw.
    float svar_inv_count;
    // factor to scale sum of squared errors to get running variance. Should be 1/nhw or 1/(nhw-1).
    float rvar_inv_count;
    // The buffer to do the reduction for mean, stddev and count.
    float *gmem_sums;
    // The buffer to count items in the different CTAs.
    int *gmem_counts;
    // The counters of retired CTAs.
    int *gmem_retired_ctas;
    // The epsilon to apply to the computation of the variance.
    float var_eps;
    // outer loop count
    int outer_loops;
    // exponential average factor
    float exp_avg_factor;
    // number of CTAs along .x dimension
    int c_blks;

    void* my_data;
    void* pair_datas[4];
    int magic;
    int sync_iters;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Storage,
    int THREADS_PER_CTA,
    int THREADS_PER_PIXEL,
    int PIXELS_PER_THREAD_IN_REGISTERS,
    int PIXELS_PER_THREAD_IN_SMEM,
    int ELEMENTS_PER_LDG,
    int USE_ONLINE_APPROACH,
    int OUTER_LOOPS_,
    bool USE_RELU,
    bool USE_ADD_RELU,
    int DESIRED_OCCUPANCY
>
__global__ __launch_bounds__(THREADS_PER_CTA, DESIRED_OCCUPANCY)
    void nhwc_batch_norm_fwd(NhwcBatchNormFwdParams params) {
    // The number of pixels loaded in a single LDG.
    const int PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
    // The number of pixels computed per CTA stored in registers.
    const int PIXELS_PER_CTA_IN_REGISTERS = PIXELS_PER_THREAD_IN_REGISTERS * PIXELS_PER_LDG;
    // The number of pixels computed per CTA stored in SMEM.
    const int PIXELS_PER_CTA_IN_SMEM = PIXELS_PER_THREAD_IN_SMEM*PIXELS_PER_LDG;
    // The number of C elements per CTA.
    const int C_ELEMENTS_PER_CTA = THREADS_PER_PIXEL*ELEMENTS_PER_LDG;

    // Shared memory to do CTA-wide parallel sums.
    __shared__ float smem[THREADS_PER_PIXEL*(THREADS_PER_CTA/32)*ELEMENTS_PER_LDG];

    // Compute the NHW coordinate of the thread in the CTA.
    const int thread_in_cta_nhw = threadIdx.x / THREADS_PER_PIXEL;

    // The adapter for the storage.
    typedef PackedStorage<Storage, ELEMENTS_PER_LDG> PackedStorage_;
    // The data type for packed storage in SMEM.
    typedef typename PackedStorage_::Type PackedStorageType;
    // The number of elements in the packed storage.
    const int PACKED_ELEMENTS_PER_LDG = PackedStorage_::PACKED_ELEMENTS_PER_LDG;
    // Registers to keep the data live for the persistent approach.
    PackedStorageType x_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];

    // Shared memory buffer to store the extra pixels.
    extern __shared__ PackedStorageType smem_storage_packed[];

    for (int c_blk_index = blockIdx.y; c_blk_index < params.c_blks; c_blk_index += gridDim.y) {
        // The position in the NHW dimension where the CTA starts.
        int cta_nhw_regs = blockIdx.x * PIXELS_PER_CTA_IN_REGISTERS;
        // The position in the NHW dimension where the CTA starts for the portion in SMEM.
        int cta_nhw_smem = blockIdx.x * PIXELS_PER_CTA_IN_SMEM;

        // The position in the C dimension where the CTA starts.
        const int cta_c = c_blk_index * C_ELEMENTS_PER_CTA;
        // Compute the C coordinate of the thread in the CTA.
        const int thread_in_cta_c = threadIdx.x % THREADS_PER_PIXEL;
        // Compute the C coordinate of the thread.
        int thread_c = cta_c + thread_in_cta_c*ELEMENTS_PER_LDG;

        // Is the thread working on a valid C dimension?
        const int is_valid_c = thread_c < params.c;

        // Clamp thread_c so that we load from valid locations even if we don't use the value
        if (!is_valid_c)
            thread_c = params.c - 4;

        // Single pass numerically stable algorithm, see:
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        //
        // n = 0, mean = 0.0, M2 = 0.0
        //
        // for x in data:
        //     n += 1
        //     delta = x - mean
        //     mean += delta/n
        //     delta2 = x - mean
        //     M2 += delta*delta2
        //
        // if n < 2:
        //     return float('nan')
        // else:
        //     return M2 / (n - 1)

        // Register to store the number of elements read so far.
        float count = 0.f, mean[ELEMENTS_PER_LDG], m2[ELEMENTS_PER_LDG];
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            mean[i] = 0.f;
            m2[i] = 0.f;
        }

        // The number of elements loaded by this CTA.
        int cta_count = 0;
        // The base pointer to load from.
        const uint16_t *gmem_src = &params.gmem_src[thread_c];

        // outer loops
        int OUTER_LOOPS = OUTER_LOOPS_ == 1? 1 : params.outer_loops;
        // Load the batch of elements. Compute the mean/var across those elements.
        const int pixels_per_iteration = PIXELS_PER_CTA_IN_REGISTERS*gridDim.x;

        if (OUTER_LOOPS_ != 1) {
            // We cannot load everything to store persistently, so let's makes sure registers and
            // smem are fully utilized, offset is evenly divisible by 32
            int offset = (pixels_per_iteration * OUTER_LOOPS +
                          PIXELS_PER_CTA_IN_SMEM * gridDim.x - params.nhw) & ~31;
            cta_nhw_regs -= offset;
            cta_nhw_smem -= offset;
        }

        #pragma unroll 1
        for (int loop_i = 0; loop_i < OUTER_LOOPS; ++loop_i) {
            // The nhw position.
            int nhw_regs = cta_nhw_regs + loop_i*pixels_per_iteration;
            // Update the number of elements loaded by this CTA. TODO: Skip if <= 0!!!
            cta_count += max(min(nhw_regs + PIXELS_PER_CTA_IN_REGISTERS, params.nhw) -
                                 max(nhw_regs, 0), 0);

            // Load the data and compute the local mean/sum and the variance.
            if (USE_ONLINE_APPROACH) {
                // Read the elements from memory.
                float is_valid[PIXELS_PER_THREAD_IN_REGISTERS];
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                    const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                    zero_array(x_storage[i]);
                    is_valid[i] = 0.f;
                    if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                        if (loop_i == OUTER_LOOPS - 1) {
                            ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                        } else {
                            ldg(x_storage[i], &gmem_src[idx*params.c]);
                        }
                        is_valid[i] = 1.f;
                    }
                }

                // Do the math.
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                    // Convert to float.
                    float x_math[ELEMENTS_PER_LDG];
                    to_float(x_math, x_storage[i]);

                    // Update the count.
                    count += is_valid[i];
                    // Invert the count.
                    float inv_count = is_valid[i] ? 1.f / count : 0.f;

                    // Update the mean and m2 using deltas.
                    #pragma unroll
                    for (int j = 0; j < ELEMENTS_PER_LDG; ++j) {
                        float delta0 = x_math[j] - mean[j];
                        mean[j] += delta0 * inv_count;
                        float delta1 = x_math[j] - mean[j];
                        m2[j] += delta0 * delta1 * is_valid[i];
                    }
                }
            } else {
                // Read the elements from memory.
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                    const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                    zero_array(x_storage[i]);
                    if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                        if (loop_i == OUTER_LOOPS - 1) {
                            ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                        } else {
                            ldg(x_storage[i], &gmem_src[idx*params.c]);
                        }
                        count += 1.f;
                    }
                }

                // Sum the elements in registers.
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                    // Convert to float.
                    float x_math[ELEMENTS_PER_LDG];
                    to_float(x_math, x_storage[i]);

                    // Update the mean and m2 using deltas.
                    #pragma unroll
                    for (int j = 0; j < ELEMENTS_PER_LDG; ++j) {
                        mean[j] += x_math[j];
                    }
                }

                // Compute the mean.
                float inv_count = 1.f / count;
                #pragma unroll
                for (int j = 0; j < ELEMENTS_PER_LDG; ++j) {
                    mean[j] *= inv_count;
                }

                // Compute the variance.
                #pragma unroll
                for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                    // Convert to float.
                    float x_math[ELEMENTS_PER_LDG];
                    to_float(x_math, x_storage[i]);

                    // Is it a valid pixel?
                    float is_valid = i < static_cast<int>(count) ? 1.f : 0.f;
                    // Update the mean and m2 using deltas.
                    #pragma unroll
                    for (int j = 0; j < ELEMENTS_PER_LDG; ++j) {
                        m2[j] += (x_math[j] - mean[j]) * (x_math[j] - mean[j]) * is_valid;
                    }
                }
            }
        }

        // The elements to load and store in SMEM.
        int smem_nhw = OUTER_LOOPS*pixels_per_iteration + cta_nhw_smem;
        // Load elements from SMEM, update the CTA count.
        int pixels_in_smem = min(smem_nhw + PIXELS_PER_CTA_IN_SMEM, params.nhw) - max(smem_nhw, 0);
        if (pixels_in_smem > 0) {
            cta_count += pixels_in_smem;
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                float is_pixel_valid = (((unsigned int)idx <
                                         (unsigned int)params.nhw) && is_valid_c) ? 1.f : 0.f;

                PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG];
                ldg_stream(x_storage_local, &gmem_src[(is_pixel_valid ? idx : 0)*params.c]);

                // The offset to store in SMEM.
                const int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                // Store in SMEM.
                write_to_smem(&smem_storage_packed[offset], threadIdx.x, x_storage_local);
                // Update the count.
                count += is_pixel_valid;
                // Invert the count.
                float inv_count = is_pixel_valid ? 1.f / count : 0.f;

                float x_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage_local);
                // Update the mean and m2 using deltas.
                #pragma unroll
                for (int j = 0; j < ELEMENTS_PER_LDG; ++j) {
                    float delta0 = x_math[j] - mean[j];
                    mean[j] += delta0 * inv_count;
                    float delta1 = x_math[j] - mean[j];
                    m2[j] += delta0 * delta1 * is_pixel_valid;
                }
            }
        }

        // We scale the mean by the number of elements. It brings more stability.
        float m1[ELEMENTS_PER_LDG];
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            m1[i] = mean[i] * count;
        }

        // Run the parallel sum accross the CTA to get the local sum.
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, m1, thread_in_cta_nhw);
        __syncthreads();

        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(m1, smem, thread_in_cta_c);
        __syncthreads();

        // Adjust the variance.
        float inv_cta_count = 1.f / static_cast<float>(cta_count);
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            float mean_diff = m1[i]*inv_cta_count - mean[i];
            m2[i] = m2[i] + mean_diff * mean_diff * count;
        }

        // Run the parallel sum accross the CTA to get the local adjusted variance.
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, m2, thread_in_cta_nhw);

        // The workspace in global memory is distributed across the different CTA.
        int gmem_sums_offset = c_blk_index*gridDim.x*C_ELEMENTS_PER_CTA*2;

        // Write the data for the CTA to global memory.
        float *gmem_sums = &params.gmem_sums[gmem_sums_offset];
        if (threadIdx.x < THREADS_PER_PIXEL) {
            const int idx = blockIdx.x*THREADS_PER_PIXEL + threadIdx.x;
            write_to_gmem(&gmem_sums[                           0], idx, m1);
            write_to_gmem(&gmem_sums[C_ELEMENTS_PER_CTA*gridDim.x], idx, m2);
        }

        // The memory location to store the number of pixels per CTA.
        int *gmem_counts = &params.gmem_counts[c_blk_index*gridDim.x];
        if (threadIdx.x == 0) {
            gmem_counts[blockIdx.x] = cta_count;
        }

        // Read the bias and scale.
        float bias[ELEMENTS_PER_LDG], scale[ELEMENTS_PER_LDG];
        if (is_valid_c) {
            read_from_gmem(bias, &params.gmem_bias[cta_c], thread_in_cta_c);
            read_from_gmem(scale, &params.gmem_scale[cta_c], thread_in_cta_c);
        }

        // The counters to count how many CTAs have retired at this point.
        // A given cta uses the same counter every other time through the outer loop.
        int *gmem_retired_ctas = &params.gmem_retired_ctas[c_blk_index % (2 * gridDim.y)];
        inter_block_sync(gmem_retired_ctas, gridDim.x, blockIdx.x == 0);

        // Reset the mean to compute the global mean.
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            m1[i] = 0.f;
        }

        // Build the global mean.
        #pragma unroll 1
        for (int idx = threadIdx.x; idx < THREADS_PER_PIXEL*gridDim.x; idx += THREADS_PER_CTA) {
            float tmp[ELEMENTS_PER_LDG];
            read_from_gmem(tmp, gmem_sums, idx);
            add(m1, tmp);
        }

        if (params.sync_iters>0)
        {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, m1, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+3, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, m1, thread_in_cta_nhw);
        }
        __syncthreads();

        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(m1, smem, thread_in_cta_c);
        __syncthreads();

        // Normalize the mean.
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            m1[i] = m1[i] * params.svar_inv_count;
        }

        // Reset the variance.
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            m2[i] = 0.f;
        }

        // for add+relu fusion
        const uint16_t *gmem_src1 = nullptr;
        if (USE_ADD_RELU) {
            gmem_src1 = &params.gmem_src1[thread_c];
        }

        // Build the global variance.
        #pragma unroll 1
        for (int idx = threadIdx.x; idx < THREADS_PER_PIXEL*gridDim.x; idx += THREADS_PER_CTA) {
            // Read the means computed by different CTAs (again). Reuse tmp if we have 1 iteration.
            float tmp_mean[ELEMENTS_PER_LDG], tmp_var[ELEMENTS_PER_LDG];
            read_from_gmem(tmp_mean, &gmem_sums[                           0], idx);
            read_from_gmem(tmp_var,  &gmem_sums[C_ELEMENTS_PER_CTA*gridDim.x], idx);

            // Read the number of pixels visited by a given CTA.
            cta_count = __ldg(&gmem_counts[idx / THREADS_PER_PIXEL]);

            // Compute the diff to update the variance.
            float mean_diff[ELEMENTS_PER_LDG], inv_cta_count = 1.f / static_cast<float>(cta_count);
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
                mean_diff[i] = m1[i] - tmp_mean[i]*inv_cta_count;
            }

            // Update the variance.
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
                m2[i] += tmp_var[i] + mean_diff[i]*mean_diff[i]*static_cast<float>(cta_count);
            }
        }

        if (params.sync_iters>0)
        {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, m2, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+2, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, m2, thread_in_cta_nhw);
        }
        __syncthreads();

        read_from_smem(m2, smem, thread_in_cta_c);

        // Finalize the stddev.
        // becasue saved var and running var may have different denominator, we don't do it here
        // scale_(m2, inv_count);

        // store the saved mean/var
        float svarinv[ELEMENTS_PER_LDG];
        bool is_valid_for_saving = is_valid_c && blockIdx.x == 0 && thread_in_cta_nhw == 0;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            svarinv[i] = rsqrtf(m2[i] * params.svar_inv_count + params.var_eps);
        }
        if (is_valid_for_saving) {
            write_to_gmem(params.gmem_saved_mean, thread_c/ELEMENTS_PER_LDG, m1);
            write_to_gmem(params.gmem_saved_var, thread_c/ELEMENTS_PER_LDG, svarinv);
        }

        // store the running mean/var
        float rmean[ELEMENTS_PER_LDG], rvar[ELEMENTS_PER_LDG];
        zero_array(rmean);
        zero_array(rvar);
        if (params.exp_avg_factor != 1.f && is_valid_for_saving) {
            read_from_gmem(rmean, params.gmem_running_mean, thread_c/ELEMENTS_PER_LDG);
            read_from_gmem(rvar, params.gmem_running_var, thread_c/ELEMENTS_PER_LDG);
        }
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            rmean[i] = (1.f - params.exp_avg_factor) * rmean[i] +   \
                params.exp_avg_factor * m1[i];
            rvar[i] = (1.f - params.exp_avg_factor) * rvar[i] +     \
                params.exp_avg_factor * (m2[i] * params.rvar_inv_count);
        }
        if (is_valid_for_saving) {
            write_to_gmem(params.gmem_running_mean, thread_c/ELEMENTS_PER_LDG, rmean);
            write_to_gmem(params.gmem_running_var, thread_c/ELEMENTS_PER_LDG, rvar);
        }

        // Update the scale with the stddev and eps.
        multiply(scale, svarinv);

        // The base pointer to write to.
        uint16_t *const gmem_dst = &params.gmem_dst[thread_c];

        unsigned int *const gmem_relu_bitmask = params.gmem_relu_bitmask +
                                     ((params.nhw + 31) & ~31) * 2 * c_blk_index;

        // Store the elements in registers.
        #pragma unroll 1
        for (int loop_i = OUTER_LOOPS-1; loop_i >= 0; --loop_i) {
            // The value for nhw.
            int out_nhw = cta_nhw_regs + loop_i*pixels_per_iteration;

            // Normalize the elements and write to memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                const bool is_valid_nhw =
                    static_cast<unsigned int>(idx) < static_cast<unsigned int>(params.nhw);
                const bool is_valid = is_valid_nhw && is_valid_c;
                // Convert to float.
                float x_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage[i]);

                // Normalize and apply activation function
                normalize(x_math, bias, scale, m1);
                if (USE_ADD_RELU) {
                    float x1_math[ELEMENTS_PER_LDG];
                    ldg_stream(x1_math, &gmem_src1[(is_valid ? idx : 0)*params.c]);
                    add(x_math, x1_math);
                    unsigned int relu_mask;
                    int lane_id = threadIdx.x & 31;
                    #pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
                        bool rectified = x_math[i] < 0.0F;
                        unsigned int local_relu_mask = __ballot_sync(0xFFFFFFFFU, rectified);
                        if (lane_id == i) {
                            // Thread 0 remembers the relu_mask from the first time through this
                            // loop, Thread 1 the next, Thread 2 the next, and Thread 3 the last.
                            relu_mask = local_relu_mask;
                        }
                        if (rectified) {
                            x_math[i] = 0.0F;
                        }
                    }
                    if (is_valid_nhw && (lane_id < ELEMENTS_PER_LDG)) {
                        gmem_relu_bitmask[idx * 2 + lane_id] = relu_mask;
                    }
                } else if (USE_RELU) {
                    relu_activation(x_math);
                }

                // Write back.
                if (is_valid) {
                    stg_stream(&gmem_dst[idx*params.c], x_math);
                }
            }

            // The next value of nhw.
            out_nhw -= pixels_per_iteration;

            // Read the next elements from memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                }
            }
        }

        // Normalize the elements from SMEM and write them out.
        if (pixels_in_smem > 0) {
            #pragma unroll 2
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                const bool is_valid_nhw =
                    static_cast<unsigned int>(idx) < static_cast<unsigned int>(params.nhw);
                const bool is_valid = is_valid_nhw && is_valid_c;

                // Read from SMEM.
                const int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG];
                read_from_smem(x_storage_local, &smem_storage_packed[offset], threadIdx.x);
                float x_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage_local);

                // Normalize and apply activation function
                normalize(x_math, bias, scale, m1);
                if (USE_ADD_RELU) {
                    float x1_math[ELEMENTS_PER_LDG];
                    ldg_stream(x1_math, &gmem_src1[(is_valid ? idx : 0)*params.c]);
                    add(x_math, x1_math);
                    unsigned int relu_mask;
                    int lane_id = threadIdx.x & 31;
                    #pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
                        bool rectified = x_math[i] < 0.0F;
                        unsigned int local_relu_mask = __ballot_sync(0xFFFFFFFFU, rectified);
                        if (lane_id == i) {
                            relu_mask = local_relu_mask;
                        }
                        if (rectified) {
                            x_math[i] = 0.0F;
                        }
                    }
                    if (is_valid_nhw && (lane_id < ELEMENTS_PER_LDG)) {
                        gmem_relu_bitmask[idx * 2 + lane_id] = relu_mask;
                    }
                } else if (USE_RELU) {
                    relu_activation(x_math);
                }

                // Write back.
                if (is_valid) {
                    stg_stream(&gmem_dst[idx*params.c], x_math);
                }
            }
        }
        // We're about to start on the next c-blk.  Needed?
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct NhwcBatchNormBwdParams {
    // The input/output tensors.
    uint16_t *gmem_src, *gmem_dy, *gmem_dst, *gmem_dst1;
    // dscale/dbias
    float *gmem_dscale, *gmem_dbias;
    // The scale and bias.
    float *gmem_scale, *gmem_bias;
    // The mean/inv-var saved from fwd pass
    float *gmem_saved_mean, *gmem_saved_var;
    // ReLU bitmask
    unsigned int *gmem_relu_bitmask;
    // The dimensions.
    int nhw, c;
    // factor to scale sum of squared errors to get saved variance.  Must be 1/nhw.
    float svar_inv_count;
    // The buffer to do the reduction for dscale and dbias
    float *gmem_sums;
    // The counters of retired CTAs.
    int *gmem_retired_ctas;
    // outer loop count
    int outer_loops;
    // number of CTAs along .x dimension
    int c_blks;

    void* my_data;
    void* pair_datas[4];
    int magic;
    int sync_iters;
    float wgrad_coeff;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void relu_bwd(float (&dy)[N], const float (&x)[N],
                              const float (&mean_var_scale_bias)[N],
                              const float (&var_scale)[N], bool valid_data) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        float y = (x[j] * var_scale[j]) + mean_var_scale_bias[j];
        if ((y <= 0.f) && valid_data) {
            dy[j] = 0.f;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void relu_bwd(float (&dy)[N], const float (&y)[N], bool valid_data) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        if ((y[j] <= 0.f) && valid_data) {
            dy[j] = 0.f;
        }
    }
}

template <int N>
DEVICE_FUNCTION void relu_bwd(float (&dy)[N], const bool (&rectified)[N], bool valid_data) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        if (rectified[j] && valid_data) {
            dy[j] = 0.f;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void relu_bwd_for_dx(float (&dy)[N],
                                     const float (&x)[N],
                                     const float (&mean_var_scale_bias)[N],
                                     const float (&var_scale)[N]) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        float y = (x[j] * var_scale[j]) + mean_var_scale_bias[j];
        if (y <= 0.f) {
            dy[j] = 0.f;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void relu_bwd_for_dx(float (&dy)[N], const float (&y)[N]) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        if (y[j] <= 0.f) {
            dy[j] = 0.f;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void bwd_update(float (&dscale)[N], float (&dbias)[N],
                                const float (&dy)[N], const float (&x)[N],
                                const float (&mean)[N], float inv_count) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        float delta0 = dy[j] - dbias[j];
        dbias[j] += delta0 * inv_count;
        delta0 = (dy[j] * (x[j] - mean[j])) - dscale[j];
        dscale[j] += delta0 * inv_count;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
DEVICE_FUNCTION void bwd_dx(float (&dx)[N], const float (&dy)[N],
                            const float (&var)[N], const float (&x)[N], const float (&mean)[N],
                            const float (&dscale)[N], const float (&dbias)[N], float inv_count) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        float tmp1 = dy[j] - (dbias[j]* inv_count);
        float tmp2 = dscale[j] * inv_count;
        float tmp3 = x[j] - mean[j];
        dx[j] = var[j] * (tmp1 - (tmp2 * tmp3));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Storage,
    int THREADS_PER_CTA,
    int THREADS_PER_PIXEL,
    int PIXELS_PER_THREAD_IN_REGISTERS,
    int PIXELS_PER_THREAD_IN_SMEM,
    int ELEMENTS_PER_LDG,
    int USE_ONLINE_APPROACH,
    int OUTER_LOOPS_,
    int DESIRED_OCCUPANCY
>
__global__ __launch_bounds__(THREADS_PER_CTA, DESIRED_OCCUPANCY)
    void nhwc_batch_norm_bwd(NhwcBatchNormBwdParams params) {
    // The number of pixels loaded in a single LDG.
    const int PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
    // The number of pixels computed per CTA stored in registers.
    const int PIXELS_PER_CTA_IN_REGISTERS = PIXELS_PER_THREAD_IN_REGISTERS * PIXELS_PER_LDG;
    // The number of pixels computed per CTA stored in SMEM.
    const int PIXELS_PER_CTA_IN_SMEM = PIXELS_PER_THREAD_IN_SMEM*PIXELS_PER_LDG;
    // The number of C elements per CTA.
    const int C_ELEMENTS_PER_CTA = THREADS_PER_PIXEL*ELEMENTS_PER_LDG;

    // Shared memory to do CTA-wide parallel sums.
    __shared__ float smem[THREADS_PER_PIXEL*(THREADS_PER_CTA/32)*ELEMENTS_PER_LDG];

    // The adapter for the storage.
    typedef PackedStorage<Storage, ELEMENTS_PER_LDG> PackedStorage_;
    // The data type for packed storage in SMEM.
    typedef typename PackedStorage_::Type PackedStorageType;
    // The number of elements in the packed storage.
    const int PACKED_ELEMENTS_PER_LDG = PackedStorage_::PACKED_ELEMENTS_PER_LDG;
    // Registers to keep the data live for the persistent approach.
    PackedStorageType x_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];
    PackedStorageType dy_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];

    // Shared memory buffer to store the extra pixels.
    extern __shared__ PackedStorageType smem_storage_packed[];

    for (int c_blk_index = blockIdx.y; c_blk_index < params.c_blks; c_blk_index += gridDim.y) {
        // The position in the NHW dimension where the CTA starts.
        int cta_nhw_regs = blockIdx.x * PIXELS_PER_CTA_IN_REGISTERS;
        // The position in the NHW dimension where the CTA starts for the portion in SMEM.
        int cta_nhw_smem = blockIdx.x * PIXELS_PER_CTA_IN_SMEM;
        // Compute the NHW coordinate of the thread in the CTA.
        const int thread_in_cta_nhw = threadIdx.x / THREADS_PER_PIXEL;

        // The position in the C dimension where the CTA starts.
        const int cta_c = c_blk_index * C_ELEMENTS_PER_CTA;
        // Compute the C coordinate of the thread in the CTA.
        const int thread_in_cta_c = threadIdx.x % THREADS_PER_PIXEL;
        // Compute the C coordinate of the thread.
        const int thread_c = cta_c + thread_in_cta_c*ELEMENTS_PER_LDG;

        // Is the thread working on a valid C dimension?
        const int is_valid_c = thread_c < params.c;

        // Registers to store the mean used for entire duration
        float mean[ELEMENTS_PER_LDG];
        zero_array(mean);
        if (is_valid_c) {
            read_from_gmem(mean, params.gmem_saved_mean, thread_c/ELEMENTS_PER_LDG);
        }

        // accumulation related registers
        float count = 0.f, dscale[ELEMENTS_PER_LDG], dbias[ELEMENTS_PER_LDG];
        zero_array(dscale);
        zero_array(dbias);

        // The number of elements loaded by this CTA.
        int cta_count = 0;
        // The base pointers to load from.
        const uint16_t *gmem_src = &params.gmem_src[thread_c];
        const uint16_t *gmem_dy = &params.gmem_dy[thread_c];

        // outer loops
        int OUTER_LOOPS = OUTER_LOOPS_ == 1? 1 : params.outer_loops;
        // Load the batch of elements. Compute sum across them
        const int pixels_per_iteration = PIXELS_PER_CTA_IN_REGISTERS*gridDim.x;

        if (OUTER_LOOPS_ != 1) {
            // We cannot load everything to store persistently, so let's makes sure registers and
            // smem are fully utilized
            int offset = params.nhw - pixels_per_iteration * OUTER_LOOPS -
                         PIXELS_PER_CTA_IN_SMEM * gridDim.x;
            cta_nhw_regs += offset;
            cta_nhw_smem += offset;
        }

        #pragma unroll 1
        for (int loop_i = 0; loop_i < OUTER_LOOPS; ++loop_i) {
            // The nhw position.
            int nhw_regs = cta_nhw_regs + loop_i*pixels_per_iteration;
            // Update the number of elements loaded by this CTA. TODO: Skip if <= 0!!!
            cta_count += max(0, min(PIXELS_PER_CTA_IN_REGISTERS, params.nhw-nhw_regs));

            // Read the elements from memory.
            float is_valid[PIXELS_PER_THREAD_IN_REGISTERS];
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                zero_array(x_storage[i]);
                zero_array(dy_storage[i]);
                is_valid[i] = 0.f;
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    if (loop_i == OUTER_LOOPS - 1) {
                        ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                        ldg_stream(dy_storage[i], &gmem_dy[idx*params.c]);
                    } else {
                        ldg(x_storage[i], &gmem_src[idx*params.c]);
                        ldg(dy_storage[i], &gmem_dy[idx*params.c]);
                    }
                    is_valid[i] = 1.f;
                }
            }

            // Do the math.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                // Convert to float and update
                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage[i]);
                to_float(dy_math, dy_storage[i]);

                // Update the count.
                count += is_valid[i];
                // Invert the count.
                float inv_count = is_valid[i] ? 1.f / count : 0.f;

                bwd_update(dscale, dbias, dy_math, x_math, mean, inv_count);
            }
        }

        // The elements to load and store in SMEM.
        int smem_nhw = OUTER_LOOPS*pixels_per_iteration + cta_nhw_smem;
        // Load elements from SMEM, update the CTA count.
        int pixels_in_smem = min(PIXELS_PER_CTA_IN_SMEM, params.nhw-smem_nhw);
        if (pixels_in_smem > 0) {
            cta_count += pixels_in_smem;
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                bool is_pixel_valid = (((unsigned int)idx <
                                        (unsigned int)params.nhw) && is_valid_c);
                PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG],
                                  dy_storage_local[PACKED_ELEMENTS_PER_LDG];
                zero_array(x_storage_local);
                zero_array(dy_storage_local);
                if (is_pixel_valid) {
                    ldg_stream(x_storage_local, &gmem_src[idx*params.c]);
                    ldg_stream(dy_storage_local, &gmem_dy[idx*params.c]);
                }

                // The offset to store in SMEM.
                int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                // Store in SMEM.
                write_to_smem(&smem_storage_packed[offset], threadIdx.x, x_storage_local);
                offset += PIXELS_PER_THREAD_IN_SMEM*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                write_to_smem(&smem_storage_packed[offset], threadIdx.x, dy_storage_local);
                // Update the count.
                count += is_pixel_valid;
                // Invert the count.
                float inv_count = is_pixel_valid ? 1.f / count : 0.f;

                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage_local);
                to_float(dy_math, dy_storage_local);

                bwd_update(dscale, dbias, dy_math, x_math, mean, inv_count);
            }
        }

        // We scale the mean by the number of elements. It brings more stability.
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            dbias[i] *= count;
            dscale[i] *= count;
        }

        // dscale parallel sum
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, dscale, thread_in_cta_nhw);
        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dscale, smem, thread_in_cta_c);
        __syncthreads();

        // dbias parallel sum
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, dbias, thread_in_cta_nhw);
        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dbias, smem, thread_in_cta_c);
        __syncthreads();

        // The workspace in global memory is distributed across the different CTA.
        int gmem_sums_offset = c_blk_index*gridDim.x*C_ELEMENTS_PER_CTA*2;
        // Write the data for the CTA to global memory.
        float *gmem_sums = &params.gmem_sums[gmem_sums_offset];
        if (threadIdx.x < THREADS_PER_PIXEL) {
            const int idx = blockIdx.x*THREADS_PER_PIXEL + threadIdx.x;
            write_to_gmem(&gmem_sums[                           0], idx, dscale);
            write_to_gmem(&gmem_sums[C_ELEMENTS_PER_CTA*gridDim.x], idx, dbias);
        }

        // The counters to count how many CTAs have retired at this point.
        // A given cta uses the same counter every other time through the outer loop.
        int *gmem_retired_ctas = &params.gmem_retired_ctas[c_blk_index % (2 * gridDim.y)];
        inter_block_sync(gmem_retired_ctas, gridDim.x, blockIdx.x == 0);

        // Reset the accumulators for global summation
        zero_array(dscale);
        zero_array(dbias);

        // Build the global accumulation
        #pragma unroll 1
        for (int idx = threadIdx.x; idx < THREADS_PER_PIXEL*gridDim.x; idx += THREADS_PER_CTA) {
            float tmp1[ELEMENTS_PER_LDG], tmp2[ELEMENTS_PER_LDG];
            read_from_gmem(tmp1, gmem_sums,                              idx);
            read_from_gmem(tmp2, gmem_sums+C_ELEMENTS_PER_CTA*gridDim.x, idx);

            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
                dscale[i] += tmp1[i];
                dbias[i] += tmp2[i];
            }
        }

        // dscale parallel sum
        if (params.sync_iters>0) {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, dscale, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+1, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, dscale, thread_in_cta_nhw);
        }

        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dscale, smem, thread_in_cta_c);
        __syncthreads();

        // dbias parallel sum
        if (params.sync_iters>0) {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, dbias, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+0, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, dbias, thread_in_cta_nhw);
        }

        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dbias, smem, thread_in_cta_c);

        // inv-var
        float var[ELEMENTS_PER_LDG];
        zero_array(var);
        if (is_valid_c) {
            read_from_gmem(var, params.gmem_saved_var, thread_c/ELEMENTS_PER_LDG);
        }

        // Normalize the dscale.
        multiply(dscale, var);

        // store dscale/dbias
        bool is_valid_for_saving = is_valid_c && blockIdx.x == 0 && thread_in_cta_nhw == 0;
        if (is_valid_for_saving) {
            if (params.sync_iters>0)
            {
                scaled_write_to_gmem(params.gmem_dscale, thread_c/ELEMENTS_PER_LDG, dscale, params.wgrad_coeff);
                scaled_write_to_gmem(params.gmem_dbias, thread_c/ELEMENTS_PER_LDG, dbias, params.wgrad_coeff);
            } else {
                write_to_gmem(params.gmem_dscale, thread_c/ELEMENTS_PER_LDG, dscale);
                write_to_gmem(params.gmem_dbias, thread_c/ELEMENTS_PER_LDG, dbias);
            }
        }

        // scale
        float scale[ELEMENTS_PER_LDG];
        zero_array(scale);
        if (is_valid_c) {
            read_from_gmem(scale, params.gmem_scale, thread_c/ELEMENTS_PER_LDG);
        }

        // Further normalize the dscale to be used in dx calculation
        multiply(dscale, var);
        // scale the inv-var as well, afterwards
        multiply(var, scale);

        // inverse count
        float inv_count = params.svar_inv_count;

        // The base pointer to write to.
        uint16_t *const gmem_dst = &params.gmem_dst[thread_c];

        // Store the elements in registers.
        #pragma unroll 1
        for (int loop_i = OUTER_LOOPS-1; loop_i >= 0; --loop_i) {
            // The value for nhw.
            int out_nhw = cta_nhw_regs + loop_i*pixels_per_iteration;

            // Normalize the elements and write to memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                // Convert to float.
                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage[i]);
                to_float(dy_math, dy_storage[i]);

                float dx[ELEMENTS_PER_LDG];
                bwd_dx(dx, dy_math, var, x_math, mean, dscale, dbias, inv_count);

                // Write back.
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    stg_stream(&gmem_dst[idx*params.c], dx);
                }
            }

            // The next value of nhw.
            out_nhw -= pixels_per_iteration;

            // Read the next elements from memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                    ldg_stream(dy_storage[i], &gmem_dy[idx*params.c]);
                }
            }
        }

        // Normalize the elements from SMEM and write them out.
        if (pixels_in_smem > 0) {
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                const bool is_valid = ((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c;
                if (is_valid) {
                    // Read from SMEM.
                    int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                    PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG],
                        dy_storage_local[PACKED_ELEMENTS_PER_LDG];
                    read_from_smem(x_storage_local, &smem_storage_packed[offset], threadIdx.x);
                    offset += PIXELS_PER_THREAD_IN_SMEM*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                    read_from_smem(dy_storage_local, &smem_storage_packed[offset], threadIdx.x);
                    float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                    to_float(x_math, x_storage_local);
                    to_float(dy_math, dy_storage_local);

                    float dx[ELEMENTS_PER_LDG];
                    bwd_dx(dx, dy_math, var, x_math, mean, dscale, dbias, inv_count);

                    // Write back.
                    stg_stream(&gmem_dst[idx*params.c], dx);
                }
            }
        }
        // We're about to start on the next c-blk.  Needed?
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Storage,
    int THREADS_PER_CTA,
    int THREADS_PER_PIXEL,
    int PIXELS_PER_THREAD_IN_REGISTERS,
    int PIXELS_PER_THREAD_IN_SMEM,
    int ELEMENTS_PER_LDG,
    int USE_ONLINE_APPROACH,
    int OUTER_LOOPS_,
    int DESIRED_OCCUPANCY
>
__global__ __launch_bounds__(THREADS_PER_CTA, DESIRED_OCCUPANCY)
    void nhwc_batch_norm_bwd_relu(NhwcBatchNormBwdParams params) {
    // The number of pixels loaded in a single LDG.
    const int PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
    // The number of pixels computed per CTA stored in registers.
    const int PIXELS_PER_CTA_IN_REGISTERS = PIXELS_PER_THREAD_IN_REGISTERS * PIXELS_PER_LDG;
    // The number of pixels computed per CTA stored in SMEM.
    const int PIXELS_PER_CTA_IN_SMEM = PIXELS_PER_THREAD_IN_SMEM*PIXELS_PER_LDG;
    // The number of C elements per CTA.
    const int C_ELEMENTS_PER_CTA = THREADS_PER_PIXEL*ELEMENTS_PER_LDG;

    // Shared memory to do CTA-wide parallel sums.
    __shared__ float smem[THREADS_PER_PIXEL*(THREADS_PER_CTA/32)*ELEMENTS_PER_LDG];

    // The adapter for the storage.
    typedef PackedStorage<Storage, ELEMENTS_PER_LDG> PackedStorage_;
    // The data type for packed storage in SMEM.
    typedef typename PackedStorage_::Type PackedStorageType;
    // The number of elements in the packed storage.
    const int PACKED_ELEMENTS_PER_LDG = PackedStorage_::PACKED_ELEMENTS_PER_LDG;
    // Registers to keep the data live for the persistent approach.
    PackedStorageType x_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];
    PackedStorageType dy_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];

    // Shared memory buffer to store the extra pixels.
    extern __shared__ PackedStorageType smem_storage_packed[];

    for (int c_blk_index = blockIdx.y; c_blk_index < params.c_blks; c_blk_index += gridDim.y) {
        // The position in the NHW dimension where the CTA starts.
        int cta_nhw_regs = blockIdx.x * PIXELS_PER_CTA_IN_REGISTERS;
        // The position in the NHW dimension where the CTA starts for the portion in SMEM.
        int cta_nhw_smem = blockIdx.x * PIXELS_PER_CTA_IN_SMEM;
        // Compute the NHW coordinate of the thread in the CTA.
        const int thread_in_cta_nhw = threadIdx.x / THREADS_PER_PIXEL;

        // The position in the C dimension where the CTA starts.
        const int cta_c = c_blk_index * C_ELEMENTS_PER_CTA;
        // Compute the C coordinate of the thread in the CTA.
        const int thread_in_cta_c = threadIdx.x % THREADS_PER_PIXEL;
        // Compute the C coordinate of the thread.
        const int thread_c = cta_c + thread_in_cta_c*ELEMENTS_PER_LDG;

        // Is the thread working on a valid C dimension?
        const int is_valid_c = thread_c < params.c;


        // Registers to store the mean/var/scale/bias used for the entire duration
        // Register usage optimizations:
        // 1. Can combine bias - (mean * var * scale) into a single register
        // 2. Can combine var * scale into a single register
        float varscale[ELEMENTS_PER_LDG];
        zero_array(varscale);
        if (is_valid_c) {
            read_from_gmem(varscale, params.gmem_saved_var, thread_c/ELEMENTS_PER_LDG);
        }
        float tmp[ELEMENTS_PER_LDG];
        zero_array(tmp);
        if (is_valid_c) {
            read_from_gmem(tmp, params.gmem_scale, thread_c/ELEMENTS_PER_LDG);
        }
        multiply(varscale, tmp);
        float mean[ELEMENTS_PER_LDG];
        zero_array(mean);
        if (is_valid_c) {
            read_from_gmem(mean, params.gmem_saved_mean, thread_c/ELEMENTS_PER_LDG);
        }
        zero_array(tmp);
        if (is_valid_c) {
            read_from_gmem(tmp, params.gmem_bias, thread_c/ELEMENTS_PER_LDG);
        }
        float mean_var_scale_bias[ELEMENTS_PER_LDG];
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            mean_var_scale_bias[i] = tmp[i] - (mean[i] * varscale[i]);
        }

        // accumulation related registers
        float count = 0.f, dscale[ELEMENTS_PER_LDG], dbias[ELEMENTS_PER_LDG];
        zero_array(dscale);
        zero_array(dbias);

        // The number of elements loaded by this CTA.
        int cta_count = 0;
        // The base pointers to load from.
        const uint16_t *gmem_src = &params.gmem_src[thread_c];
        const uint16_t *gmem_dy = &params.gmem_dy[thread_c];

        // outer loops
        int OUTER_LOOPS = OUTER_LOOPS_ == 1? 1 : params.outer_loops;
        // Load the batch of elements. Compute sum across them
        const int pixels_per_iteration = PIXELS_PER_CTA_IN_REGISTERS*gridDim.x;

        if (OUTER_LOOPS_ != 1) {
            // We cannot load everything to store persistently, so let's makes sure registers and
            // smem are fully utilized
            int offset = params.nhw - pixels_per_iteration * OUTER_LOOPS -
                         PIXELS_PER_CTA_IN_SMEM * gridDim.x;
            cta_nhw_regs += offset;
            cta_nhw_smem += offset;
        }

        #pragma unroll 1
        for (int loop_i = 0; loop_i < OUTER_LOOPS; ++loop_i) {
            // The nhw position.
            int nhw_regs = cta_nhw_regs + loop_i*pixels_per_iteration;
            // Update the number of elements loaded by this CTA. TODO: Skip if <= 0!!!
            cta_count += max(0, min(PIXELS_PER_CTA_IN_REGISTERS, params.nhw-nhw_regs));

            // Read the elements from memory.
            float is_valid[PIXELS_PER_THREAD_IN_REGISTERS];
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                zero_array(x_storage[i]);
                zero_array(dy_storage[i]);
                is_valid[i] = 0.f;
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    if (loop_i == OUTER_LOOPS - 1) {
                        ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                        ldg_stream(dy_storage[i], &gmem_dy[idx*params.c]);
                    } else {
                        ldg(x_storage[i], &gmem_src[idx*params.c]);
                        ldg(dy_storage[i], &gmem_dy[idx*params.c]);
                    }
                    is_valid[i] = 1.f;
                }
            }

            // Do the math.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                // Convert to float and update
                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage[i]);
                to_float(dy_math, dy_storage[i]);

                // Update the count.
                count += is_valid[i];
                // Invert the count.
                float inv_count = is_valid[i] ? 1.f / count : 0.f;

                relu_bwd(dy_math, x_math, mean_var_scale_bias, varscale, is_valid[i]);
                bwd_update(dscale, dbias, dy_math, x_math, mean, inv_count);
            }
        }

        // The elements to load and store in SMEM.
        int smem_nhw = OUTER_LOOPS*pixels_per_iteration + cta_nhw_smem;
        // Load elements from SMEM, update the CTA count.
        int pixels_in_smem = min(PIXELS_PER_CTA_IN_SMEM, params.nhw-smem_nhw);
        if (pixels_in_smem > 0) {
            cta_count += pixels_in_smem;
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                bool is_pixel_valid = (((unsigned int)idx <
                                        (unsigned int)params.nhw) && is_valid_c);
                PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG],
                                  dy_storage_local[PACKED_ELEMENTS_PER_LDG];
                zero_array(x_storage_local);
                zero_array(dy_storage_local);
                if (is_pixel_valid) {
                    ldg_stream(x_storage_local, &gmem_src[idx*params.c]);
                    ldg_stream(dy_storage_local, &gmem_dy[idx*params.c]);
                }

                // The offset to store in SMEM.
                int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                // Store in SMEM.
                write_to_smem(&smem_storage_packed[offset], threadIdx.x, x_storage_local);
                offset += PIXELS_PER_THREAD_IN_SMEM*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                write_to_smem(&smem_storage_packed[offset], threadIdx.x, dy_storage_local);
                // Update the count.
                count += is_pixel_valid;
                // Invert the count.
                float inv_count = is_pixel_valid ? 1.f / count : 0.f;

                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage_local);
                to_float(dy_math, dy_storage_local);

                relu_bwd(dy_math, x_math, mean_var_scale_bias, varscale, is_pixel_valid);
                bwd_update(dscale, dbias, dy_math, x_math, mean, inv_count);
            }
        }

        // We scale the mean by the number of elements. It brings more stability.
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            dbias[i] *= count;
            dscale[i] *= count;
        }

        // dscale parallel sum
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, dscale, thread_in_cta_nhw);
        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dscale, smem, thread_in_cta_c);
        __syncthreads();

        // dbias parallel sum
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, dbias, thread_in_cta_nhw);
        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dbias, smem, thread_in_cta_c);
        __syncthreads();

        // The workspace in global memory is distributed across the different CTA.
        int gmem_sums_offset = c_blk_index*gridDim.x*C_ELEMENTS_PER_CTA*2;
        // Write the data for the CTA to global memory.
        float *gmem_sums = &params.gmem_sums[gmem_sums_offset];
        if (threadIdx.x < THREADS_PER_PIXEL) {
            const int idx = blockIdx.x*THREADS_PER_PIXEL + threadIdx.x;
            write_to_gmem(&gmem_sums[                           0], idx, dscale);
            write_to_gmem(&gmem_sums[C_ELEMENTS_PER_CTA*gridDim.x], idx, dbias);
        }

        // The counters to count how many CTAs have retired at this point.
        // A given cta uses the same counter every other time through the outer loop.
        int *gmem_retired_ctas = &params.gmem_retired_ctas[c_blk_index % (2 * gridDim.y)];
        inter_block_sync(gmem_retired_ctas, gridDim.x, blockIdx.x == 0);

        // Reset the accumulators for global summation
        zero_array(dscale);
        zero_array(dbias);

        // Build the global accumulation
        #pragma unroll 1
        for (int idx = threadIdx.x; idx < THREADS_PER_PIXEL*gridDim.x; idx += THREADS_PER_CTA) {
            float tmp1[ELEMENTS_PER_LDG], tmp2[ELEMENTS_PER_LDG];
            read_from_gmem(tmp1, gmem_sums,                              idx);
            read_from_gmem(tmp2, gmem_sums+C_ELEMENTS_PER_CTA*gridDim.x, idx);

            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
                dscale[i] += tmp1[i];
                dbias[i] += tmp2[i];
            }
        }

        // dscale parallel sum
        if (params.sync_iters>0) {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, dscale, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+1, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, dscale, thread_in_cta_nhw);
        }

        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dscale, smem, thread_in_cta_c);
        __syncthreads();

        // dbias parallel sum
        if (params.sync_iters>0) {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, dbias, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+0, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, dbias, thread_in_cta_nhw);
        }

        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dbias, smem, thread_in_cta_c);

        // Normalize the dscale.
        float var[ELEMENTS_PER_LDG];
        zero_array(var);
        if (is_valid_c) {
            read_from_gmem(var, params.gmem_saved_var, thread_c/ELEMENTS_PER_LDG);
        }
        multiply(dscale, var);

        // store dscale/dbias
        bool is_valid_for_saving = is_valid_c && blockIdx.x == 0 && thread_in_cta_nhw == 0;
        if (is_valid_for_saving) {
            if (params.sync_iters>0)
            {
                scaled_write_to_gmem(params.gmem_dscale, thread_c/ELEMENTS_PER_LDG, dscale, params.wgrad_coeff);
                scaled_write_to_gmem(params.gmem_dbias, thread_c/ELEMENTS_PER_LDG, dbias, params.wgrad_coeff);
            } else {
                write_to_gmem(params.gmem_dscale, thread_c/ELEMENTS_PER_LDG, dscale);
                write_to_gmem(params.gmem_dbias, thread_c/ELEMENTS_PER_LDG, dbias);
            }
        }

        // Further normalize the dscale to be used in dx calculation
        float scale[ELEMENTS_PER_LDG];
        zero_array(scale);
        if (is_valid_c) {
            read_from_gmem(scale, params.gmem_scale, thread_c/ELEMENTS_PER_LDG);
        }
        multiply(dscale, var);
        // scale the inv-var as well, afterwards
        multiply(var, scale);

        // inverse count
        float inv_count = params.svar_inv_count;

        // The base pointer to write to.
        uint16_t *const gmem_dst = &params.gmem_dst[thread_c];

        // Store the elements in registers.
        #pragma unroll 1
        for (int loop_i = OUTER_LOOPS-1; loop_i >= 0; --loop_i) {
            // The value for nhw.
            int out_nhw = cta_nhw_regs + loop_i*pixels_per_iteration;

            // Normalize the elements and write to memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                // Convert to float.
                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage[i]);
                to_float(dy_math, dy_storage[i]);
                relu_bwd_for_dx(dy_math, x_math, mean_var_scale_bias, var);

                float dx[ELEMENTS_PER_LDG];
                bwd_dx(dx, dy_math, var, x_math, mean, dscale, dbias, inv_count);

                // Write back.
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    stg_stream(&gmem_dst[idx*params.c], dx);
                }
            }

            // The next value of nhw.
            out_nhw -= pixels_per_iteration;

            // Read the next elements from memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                    ldg_stream(dy_storage[i], &gmem_dy[idx*params.c]);
                }
            }
        }

        // Normalize the elements from SMEM and write them out.
        if (pixels_in_smem > 0) {
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                const bool is_valid = ((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c;
                if (is_valid) {
                    // Read from SMEM.
                    int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                    PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG],
                        dy_storage_local[PACKED_ELEMENTS_PER_LDG];
                    read_from_smem(x_storage_local, &smem_storage_packed[offset], threadIdx.x);
                    offset += PIXELS_PER_THREAD_IN_SMEM*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                    read_from_smem(dy_storage_local, &smem_storage_packed[offset], threadIdx.x);
                    float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                    to_float(x_math, x_storage_local);
                    to_float(dy_math, dy_storage_local);
                    relu_bwd_for_dx(dy_math, x_math, mean_var_scale_bias, var);

                    float dx[ELEMENTS_PER_LDG];
                    bwd_dx(dx, dy_math, var, x_math, mean, dscale, dbias, inv_count);

                    // Write back.
                    stg_stream(&gmem_dst[idx*params.c], dx);
                }
            }
        }
        // We're about to start on the next c-blk.  Needed?
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Storage,
    int THREADS_PER_CTA,
    int THREADS_PER_PIXEL,
    int PIXELS_PER_THREAD_IN_REGISTERS,
    int PIXELS_PER_THREAD_IN_SMEM,
    int ELEMENTS_PER_LDG,
    int USE_ONLINE_APPROACH,
    int OUTER_LOOPS_,
    int DESIRED_OCCUPANCY
>
__global__ __launch_bounds__(THREADS_PER_CTA, DESIRED_OCCUPANCY)
    void nhwc_batch_norm_bwd_add_relu(NhwcBatchNormBwdParams params) {
    // The number of pixels loaded in a single LDG.
    const int PIXELS_PER_LDG = THREADS_PER_CTA / THREADS_PER_PIXEL;
    // The number of pixels computed per CTA stored in registers.
    const int PIXELS_PER_CTA_IN_REGISTERS = PIXELS_PER_THREAD_IN_REGISTERS * PIXELS_PER_LDG;
    // The number of pixels computed per CTA stored in SMEM.
    const int PIXELS_PER_CTA_IN_SMEM = PIXELS_PER_THREAD_IN_SMEM*PIXELS_PER_LDG;
    // The number of C elements per CTA.
    const int C_ELEMENTS_PER_CTA = THREADS_PER_PIXEL*ELEMENTS_PER_LDG;

    // Shared memory to do CTA-wide parallel sums.
    __shared__ float smem[THREADS_PER_PIXEL*(THREADS_PER_CTA/32)*ELEMENTS_PER_LDG];

    // The adapter for the storage.
    typedef PackedStorage<Storage, ELEMENTS_PER_LDG> PackedStorage_;
    // The data type for packed storage in SMEM.
    typedef typename PackedStorage_::Type PackedStorageType;
    // The number of elements in the packed storage.
    const int PACKED_ELEMENTS_PER_LDG = PackedStorage_::PACKED_ELEMENTS_PER_LDG;
    // Registers to keep the data live for the persistent approach.
    PackedStorageType x_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];
    PackedStorageType dy_storage[PIXELS_PER_THREAD_IN_REGISTERS][PACKED_ELEMENTS_PER_LDG];

    // Shared memory buffer to store the extra pixels.
    extern __shared__ PackedStorageType smem_storage_packed[];

    for (int c_blk_index = blockIdx.y; c_blk_index < params.c_blks; c_blk_index += gridDim.y) {
        // The position in the NHW dimension where the CTA starts.
        int cta_nhw_regs = blockIdx.x * PIXELS_PER_CTA_IN_REGISTERS;
        // The position in the NHW dimension where the CTA starts for the portion in SMEM.
        int cta_nhw_smem = blockIdx.x * PIXELS_PER_CTA_IN_SMEM;
        // Compute the NHW coordinate of the thread in the CTA.
        const int thread_in_cta_nhw = threadIdx.x / THREADS_PER_PIXEL;

        // The position in the C dimension where the CTA starts.
        const int cta_c = c_blk_index * C_ELEMENTS_PER_CTA;
        // Compute the C coordinate of the thread in the CTA.
        const int thread_in_cta_c = threadIdx.x % THREADS_PER_PIXEL;
        // Compute the C coordinate of the thread.
        const int thread_c = cta_c + thread_in_cta_c*ELEMENTS_PER_LDG;

        // Is the thread working on a valid C dimension?
        const int is_valid_c = thread_c < params.c;

        float mean[ELEMENTS_PER_LDG];
        zero_array(mean);
        if (is_valid_c) {
            read_from_gmem(mean, params.gmem_saved_mean, thread_c/ELEMENTS_PER_LDG);
        }

        // accumulation related registers
        float count = 0.f, dscale[ELEMENTS_PER_LDG], dbias[ELEMENTS_PER_LDG];
        zero_array(dscale);
        zero_array(dbias);

        // The number of elements loaded by this CTA.
        int cta_count = 0;
        // The base pointers to load from.
        const uint16_t *gmem_src = &params.gmem_src[thread_c];
        const uint16_t *gmem_dy = &params.gmem_dy[thread_c];
        uint16_t *gmem_dst1 = &params.gmem_dst1[thread_c];

        // outer loops
        int OUTER_LOOPS = OUTER_LOOPS_ == 1? 1 : params.outer_loops;
        // Load the batch of elements. Compute sum across them
        const int pixels_per_iteration = PIXELS_PER_CTA_IN_REGISTERS*gridDim.x;

        if (OUTER_LOOPS_ != 1) {
            // We cannot load everything to store persistently, so let's makes sure registers and
            // smem are fully utilized, offset is evenly divisible by 32
            int offset = (pixels_per_iteration * OUTER_LOOPS + PIXELS_PER_CTA_IN_SMEM * gridDim.x -
                          params.nhw) & ~31;
            cta_nhw_regs -= offset;
            cta_nhw_smem -= offset;
        }

        const unsigned int *const gmem_relu_bitmask = params.gmem_relu_bitmask +
                                      ((params.nhw + 31) & ~31) * 2 * c_blk_index;

        #pragma unroll 1
        for (int loop_i = 0; loop_i < OUTER_LOOPS; ++loop_i) {
            // The nhw position.
            int nhw_regs = cta_nhw_regs + loop_i*pixels_per_iteration;
            // Update the number of elements loaded by this CTA. TODO: Skip if <= 0!!!
            cta_count += max(0, min(PIXELS_PER_CTA_IN_REGISTERS, params.nhw-nhw_regs));

            int lane_id = threadIdx.x & 31;

            // Read the elements from memory.
            float is_valid[PIXELS_PER_THREAD_IN_REGISTERS];
            unsigned int relu_mask[PIXELS_PER_THREAD_IN_REGISTERS];
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                zero_array(x_storage[i]);
                zero_array(dy_storage[i]);
                is_valid[i] = 0.f;
                const bool is_valid_nhw =
                    static_cast<unsigned int>(idx) < static_cast<unsigned int>(params.nhw);
                if (is_valid_nhw) {
                    if (is_valid_c) {
                        if (loop_i == OUTER_LOOPS - 1) {
                            ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                            ldg_stream(dy_storage[i], &gmem_dy[idx*params.c]);
                        } else {
                            ldg(x_storage[i], &gmem_src[idx*params.c]);
                            ldg(dy_storage[i], &gmem_dy[idx*params.c]);
                        }
                        is_valid[i] = 1.f;
                    }

                    if (lane_id < ELEMENTS_PER_LDG) {
                        relu_mask[i] = gmem_relu_bitmask[idx * 2 + lane_id];
                    }
                }
            }

            // Do the math.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = nhw_regs + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                // Convert to float and update
                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                bool rectified[ELEMENTS_PER_LDG];
                #pragma unroll
                for (int j = 0; j < ELEMENTS_PER_LDG; ++j) {
                    rectified[j] = ((__shfl_sync(0xFFFFFFFFU, relu_mask[i], j) &
                                    (1U << lane_id)) != 0);
                }
                to_float(x_math, x_storage[i]);
                to_float(dy_math, dy_storage[i]);

                // Update the count.
                count += is_valid[i];
                // Invert the count.
                float inv_count = is_valid[i] ? 1.f / count : 0.f;

                relu_bwd(dy_math, rectified, is_valid[i]);
                bwd_update(dscale, dbias, dy_math, x_math, mean, inv_count);

                // Lastly we need 'dy' only for BN, so store the 'relu-dgrad'ed version
                from_float(dy_storage[i], dy_math);

                // dZ for elementwise add
                if (is_valid[i]) {
                    if (loop_i == OUTER_LOOPS - 1) {
                        stg_stream(&gmem_dst1[idx*params.c], dy_storage[i]);
                    } else {
                        stg(&gmem_dst1[idx*params.c], dy_storage[i]);
                    }
                }
            }
        }

        // The elements to load and store in SMEM.
        int smem_nhw = OUTER_LOOPS*pixels_per_iteration + cta_nhw_smem;
        // Load elements from SMEM, update the CTA count.
        int pixels_in_smem = min(PIXELS_PER_CTA_IN_SMEM, params.nhw-smem_nhw);
        if (pixels_in_smem > 0) {
            cta_count += pixels_in_smem;
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                const bool is_pixel_valid_nhw =
                    static_cast<unsigned int>(idx) < static_cast<unsigned int>(params.nhw);
                const bool is_pixel_valid = is_pixel_valid_nhw && is_valid_c;
                PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG],
                                  dy_storage_local[PACKED_ELEMENTS_PER_LDG];
                unsigned int relu_mask;
                int lane_id = threadIdx.x & 31;
                zero_array(x_storage_local);
                zero_array(dy_storage_local);
                if (is_pixel_valid_nhw) {
                    if (is_valid_c) {
                        ldg_stream(x_storage_local, &gmem_src[idx*params.c]);
                        ldg_stream(dy_storage_local, &gmem_dy[idx*params.c]);
                    }
                    if (lane_id < ELEMENTS_PER_LDG) {
                        relu_mask = gmem_relu_bitmask[idx * 2 + lane_id];
                    }
                }
                bool rectified[ELEMENTS_PER_LDG];
                #pragma unroll
                for (int j = 0; j < ELEMENTS_PER_LDG; ++j) {
                    rectified[j] = ((__shfl_sync(0xFFFFFFFFU, relu_mask, j) &
                                    (1U << lane_id)) != 0);
                }

                // The offset to store in SMEM.
                int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                // Store in SMEM.
                write_to_smem(&smem_storage_packed[offset], threadIdx.x, x_storage_local);
                offset += PIXELS_PER_THREAD_IN_SMEM*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                // Update the count.
                count += is_pixel_valid;
                // Invert the count.
                float inv_count = is_pixel_valid ? 1.f / count : 0.f;

                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage_local);
                to_float(dy_math, dy_storage_local);

                relu_bwd(dy_math, rectified, is_pixel_valid);
                bwd_update(dscale, dbias, dy_math, x_math, mean, inv_count);

                from_float(dy_storage_local, dy_math);
                // dZ for elementwise add
                if (is_pixel_valid) {
                    stg_stream(&gmem_dst1[idx*params.c], dy_storage_local);
                }
                // only store the 'relu-dgrad'ed version!
                write_to_smem(&smem_storage_packed[offset], threadIdx.x, dy_storage_local);
            }
        }

        // We scale the mean by the number of elements. It brings more stability.
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
            dbias[i] *= count;
            dscale[i] *= count;
        }

        // dscale parallel sum
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, dscale, thread_in_cta_nhw);
        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dscale, smem, thread_in_cta_c);
        __syncthreads();

        // dbias parallel sum
        ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
            smem, dbias, thread_in_cta_nhw);
        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dbias, smem, thread_in_cta_c);
        __syncthreads();

        // The workspace in global memory is distributed across the different CTA.
        int gmem_sums_offset = c_blk_index*gridDim.x*C_ELEMENTS_PER_CTA*2;
        // Write the data for the CTA to global memory.
        float *gmem_sums = &params.gmem_sums[gmem_sums_offset];
        if (threadIdx.x < THREADS_PER_PIXEL) {
            const int idx = blockIdx.x*THREADS_PER_PIXEL + threadIdx.x;
            write_to_gmem(&gmem_sums[                           0], idx, dscale);
            write_to_gmem(&gmem_sums[C_ELEMENTS_PER_CTA*gridDim.x], idx, dbias);
        }

        // The counters to count how many CTAs have retired at this point.
        // A given cta uses the same counter every other time through the outer loop.
        int *gmem_retired_ctas = &params.gmem_retired_ctas[c_blk_index % (2 * gridDim.y)];
        inter_block_sync(gmem_retired_ctas, gridDim.x, blockIdx.x == 0);

        // Reset the accumulators for global summation
        zero_array(dscale);
        zero_array(dbias);

        // Build the global accumulation
        #pragma unroll 1
        for (int idx = threadIdx.x; idx < THREADS_PER_PIXEL*gridDim.x; idx += THREADS_PER_CTA) {
            float tmp1[ELEMENTS_PER_LDG], tmp2[ELEMENTS_PER_LDG];
            read_from_gmem(tmp1, gmem_sums,                              idx);
            read_from_gmem(tmp2, gmem_sums+C_ELEMENTS_PER_CTA*gridDim.x, idx);

            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LDG; ++i) {
                dscale[i] += tmp1[i];
                dbias[i] += tmp2[i];
            }
        }

        // dscale parallel sum
        if (params.sync_iters>0) {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, dscale, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+1, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, dscale, thread_in_cta_nhw);
        }

        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dscale, smem, thread_in_cta_c);
        __syncthreads();

        // dbias parallel sum
        if (params.sync_iters>0) {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatchX<THREADS_PER_CTA>(
                smem, dbias, thread_in_cta_nhw, params.my_data, params.pair_datas, 4*c_blk_index+0, params.magic, params.sync_iters);
        } else {
            ParallelSums<THREADS_PER_PIXEL, ELEMENTS_PER_LDG>::dispatch<THREADS_PER_CTA>(
                smem, dbias, thread_in_cta_nhw);
        }

        __syncthreads();
        // The values in shared memory correspond to the CTA-wide sums.
        read_from_smem(dbias, smem, thread_in_cta_c);

        // Normalize the dscale.
        float var[ELEMENTS_PER_LDG];
        zero_array(var);
        if (is_valid_c) {
            read_from_gmem(var, params.gmem_saved_var, thread_c/ELEMENTS_PER_LDG);
        }
        multiply(dscale, var);

        // store dscale/dbias
        bool is_valid_for_saving = is_valid_c && blockIdx.x == 0 && thread_in_cta_nhw == 0;
        if (is_valid_for_saving) {
            if (params.sync_iters>0)
            {
                scaled_write_to_gmem(params.gmem_dscale, thread_c/ELEMENTS_PER_LDG, dscale, params.wgrad_coeff);
                scaled_write_to_gmem(params.gmem_dbias, thread_c/ELEMENTS_PER_LDG, dbias, params.wgrad_coeff);
            } else {
                write_to_gmem(params.gmem_dscale, thread_c/ELEMENTS_PER_LDG, dscale);
                write_to_gmem(params.gmem_dbias, thread_c/ELEMENTS_PER_LDG, dbias);
            }
        }

        // Further normalize the dscale to be used in dx calculation
        float scale[ELEMENTS_PER_LDG];
        zero_array(scale);
        if (is_valid_c) {
            read_from_gmem(scale, params.gmem_scale, thread_c/ELEMENTS_PER_LDG);
        }
        multiply(dscale, var);
        // scale the inv-var as well, afterwards
        multiply(var, scale);

        // inverse count
        float inv_count = params.svar_inv_count;

        // The base pointer to write to.
        uint16_t *const gmem_dst = &params.gmem_dst[thread_c];

        // Store the elements in registers.
        #pragma unroll 1
        for (int loop_i = OUTER_LOOPS-1; loop_i >= 0; --loop_i) {
            // The value for nhw.
            int out_nhw = cta_nhw_regs + loop_i*pixels_per_iteration;

            // Normalize the elements and write to memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                const bool is_valid = ((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c;
                // Convert to float.
                float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                to_float(x_math, x_storage[i]);
                to_float(dy_math, dy_storage[i]);

                float dx[ELEMENTS_PER_LDG];
                bwd_dx(dx, dy_math, var, x_math, mean, dscale, dbias, inv_count);

                // Write back.
                if (is_valid) {
                    stg_stream(&gmem_dst[idx*params.c], dx);
                }
            }

            // The next value of nhw.
            out_nhw -= pixels_per_iteration;

            // Read the next elements from memory.
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_IN_REGISTERS; ++i) {
                const int idx = out_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                float y[ELEMENTS_PER_LDG];
                zero_array(y);
                if (((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c) {
                    ldg_stream(x_storage[i], &gmem_src[idx*params.c]);
                    ldg_stream(dy_storage[i], &gmem_dst1[idx*params.c]);
                }
            }
        }

        // Normalize the elements from SMEM and write them out.
        if (pixels_in_smem > 0) {
            for (int i = 0; i < PIXELS_PER_THREAD_IN_SMEM; ++i) {
                const int idx = smem_nhw + thread_in_cta_nhw + i*PIXELS_PER_LDG;
                const bool is_valid = ((unsigned int)idx < (unsigned int)params.nhw) && is_valid_c;
                if (is_valid) {
                    // Read from SMEM.
                    int offset = i*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                    PackedStorageType x_storage_local[PACKED_ELEMENTS_PER_LDG],
                        dy_storage_local[PACKED_ELEMENTS_PER_LDG];
                    read_from_smem(x_storage_local, &smem_storage_packed[offset], threadIdx.x);
                    offset += PIXELS_PER_THREAD_IN_SMEM*THREADS_PER_CTA*PACKED_ELEMENTS_PER_LDG;
                    read_from_smem(dy_storage_local, &smem_storage_packed[offset], threadIdx.x);
                    float x_math[ELEMENTS_PER_LDG], dy_math[ELEMENTS_PER_LDG];
                    to_float(x_math, x_storage_local);
                    to_float(dy_math, dy_storage_local);

                    float dx[ELEMENTS_PER_LDG];
                    bwd_dx(dx, dy_math, var, x_math, mean, dscale, dbias, inv_count);

                    // Write back.
                    stg_stream(&gmem_dst[idx*params.c], dx);
                }
            }
        }
        // We're about to start on the next c-blk.  Needed?
        __syncthreads();
    }
}

#endif  // MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_KERNEL_H_
