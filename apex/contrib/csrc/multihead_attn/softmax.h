#pragma once
 
#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cmath>
 
namespace {
    template <typename Datatype, int ELEMENTS_PER_LDG>
    __device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);
 
    template <>
    __device__ __inline__ void copy_vector<__half, 1>(__half *dst, const __half *src) { *dst = *src; }
 
    template <>
    __device__ __inline__ void copy_vector<float, 1>(float *dst, const float *src) { *dst = *src; }
 
    template <typename Datatype, int ELEMENTS_PER_LDG>
    __device__ __inline__ void apply_mask(Datatype *dst, Datatype value, const uint8_t *src);
    
    template <>
    __device__ __inline__ void apply_mask<__half, 1>(__half *dst, __half value, const uint8_t *src) {
      if (*src == 1) { *dst = value; }
    }
 
} // namespace anonymous
 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp Softmax forward
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG=1>
__global__ void softmax_warp_forward(input_t *dst, const output_t *src, int batch_size, int stride, int element_count)
{
    assert(ELEMENTS_PER_LDG_STG==1);
 
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
 
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;
 
    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
 
    src += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    dst += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
 
    // load data from global memory
    input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            #pragma unroll
            for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                elements_input[i][it + element] = -std::numeric_limits<float>::infinity();
            }
 
            if (element_index < batch_element_count) {
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], src + i * element_count + it * WARP_SIZE);
            }
 
        }
    }
 
    // convert input_t to acc_t
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = elements_input[i][it];
        }
    }
 
    constexpr uint32_t  FULL_MASK = 0xffffffff;
 
    // compute local max_value
 
    // take the max_value of the first element to avoid one max call
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        max_value[i] = elements[i][0];
    }
 
    #pragma unroll
    for (int it = 1;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
 
    // reduction max_value
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float val[WARP_BATCH];
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
        }
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }
 
    // compute local sum
    acc_t sum[WARP_BATCH] { 0.0f };
 
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            //elements[i][it] = expf(elements[i][it] - max_value[i]);
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }
 
    // reduction sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }
 
    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                //dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
                output_t out[ELEMENTS_PER_LDG_STG];
                for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                    out[element] = elements[i][it + element] / sum[i];
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + i * element_count + it * WARP_SIZE, out);
            }
            else {
                break;
            }
        }
    }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using softmax_forward_func = void(*)(input_t *dst, const output_t *src, int batch_size, int stride, int element_count);
 
template <typename input_t, typename output_t, typename acc_t>
bool warp_softmax_kernel(int log2_elements, int &warp_size, int &batches_per_warp, softmax_forward_func<input_t, output_t> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;
 
    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
 
    switch (log2_elements) {
    case 0: // 1
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,1,1>;
        break;
    case 1: // 2
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,2,1>;
        break;
    case 2: // 4
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,4,1>;
        break;
    case 3: // 8
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,8,1>;
        break;
    case 4: // 16
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,16,1>;
        break;
    case 5: // 32
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,32,1>;
        break;
    case 6: // 64
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,2,32,1>;
        break;
    case 7: // 128
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,4,32,1>;
        break;
    case 8: // 256
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1,8,32,1>;
        break;
    case 9: // 512
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1,16,32,1>;
        break;
    case 10: // 1024
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1,32,32,1>;
        break;
    default:
        return false;
    }
    return true;
}

template<typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax(output_t *dst, const input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 1024) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;
 
        softmax_forward_func<input_t, output_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_softmax_kernel<input_t, output_t, acc_t>(log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }
 
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
 
        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);
 
        // compute launch size
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
 
        // launch
        kernel<<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        return true;
    }
    return false;
}


// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG=1>
__global__ void masked_softmax_warp_forward(input_t *dst, const output_t *src, const uint8_t *pad_mask, int batch_size, int stride, int element_count, int pad_batch_stride)
{
    assert(ELEMENTS_PER_LDG_STG==1);
 
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
 
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;
 
    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    int thread_offset =  first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    src += thread_offset;
    dst += thread_offset;
 
    // load data from global memory
    input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        int pad_thread_offset = ( (first_batch + i) / pad_batch_stride) * stride + ELEMENTS_PER_LDG_STG * local_idx;
        const uint8_t* curr_mask    = pad_mask + pad_thread_offset;
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            #pragma unroll
            for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                elements_input[i][it + element] = -std::numeric_limits<float>::infinity();
            }
 
            if (element_index < batch_element_count) {
                int itr_jmp = it * WARP_SIZE;
                int itr_idx = i * element_count + itr_jmp;
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], src + itr_idx);
                apply_mask<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], 
                                                          (__half)-std::numeric_limits<float>::infinity(), 
                                                          curr_mask + itr_jmp);
            }
 
        }
    }
 
    // convert input_t to acc_t
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = elements_input[i][it];
        }
    }
 
    constexpr uint32_t  FULL_MASK = 0xffffffff;
 
    // compute local max_value
 
    // take the max_value of the first element to avoid one max call
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        max_value[i] = elements[i][0];
    }
 
    #pragma unroll
    for (int it = 1;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
 
    // reduction max_value
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float val[WARP_BATCH];
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
        }
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }
 
    // compute local sum
    acc_t sum[WARP_BATCH] { 0.0f };
 
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            //elements[i][it] = expf(elements[i][it] - max_value[i]);
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }
 
    // reduction sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }
 
    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                //dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
                output_t out[ELEMENTS_PER_LDG_STG];
                for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                    out[element] = elements[i][it + element] / sum[i];
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + i * element_count + it * WARP_SIZE, out);
            }
            else {
                break;
            }
        }
    }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using masked_softmax_forward_func = void(*)(input_t *dst, const output_t *src, const uint8_t *pad_mask, int batch_size, int stride, int element_count, int pad_batch_stride);
 
template <typename input_t, typename output_t, typename acc_t>
bool warp_masked_softmax_kernel(int log2_elements, int &warp_size, int &batches_per_warp, masked_softmax_forward_func<input_t, output_t> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;
 
    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
 
    switch (log2_elements) {
    case 0: // 1
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,1,1>;
        break;
    case 1: // 2
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,2,1>;
        break;
    case 2: // 4
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,4,1>;
        break;
    case 3: // 8
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,8,1>;
        break;
    case 4: // 16
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,16,1>;
        break;
    case 5: // 32
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,32,1>;
        break;
    case 6: // 64
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,2,32,1>;
        break;
    case 7: // 128
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2,4,32,1>;
        break;
    case 8: // 256
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 1,8,32,1>;
        break;
    case 9: // 512
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 1,16,32,1>;
        break;
    case 10: // 1024
        kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 1,32,32,1>;
        break;
    default:
        return false;
    }
    return true;
}
 
template<typename input_t, typename output_t, typename acc_t>
bool dispatch_masked_softmax(output_t *dst, const input_t *src, const uint8_t *pad_mask, int softmax_elements, int softmax_elements_stride, int batch_count, int pad_batch_stride)
{
    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 1024) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;
 
        masked_softmax_forward_func<input_t, output_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_masked_softmax_kernel<input_t, output_t, acc_t>(log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }
 
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
 
        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);
 
        // compute launch size
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
 
        // launch
        kernel<<<blocks, threads>>>(dst, src, pad_mask, batch_count, softmax_elements_stride, softmax_elements, pad_batch_stride);
        return true;
    }
    return false;
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG=1>
__global__ void time_masked_softmax_warp_forward(input_t *dst, const output_t *src, const uint8_t *pad_mask, int batch_size, int stride, int element_count, int mod_seq_len)
{
    assert(ELEMENTS_PER_LDG_STG==1);
 
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
 
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;
 
    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    int thread_offset =  first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    src += thread_offset;
    dst += thread_offset;
 
    // load data from global memory
    input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        int pad_thread_offset = ( (first_batch + i) % mod_seq_len) * stride + ELEMENTS_PER_LDG_STG * local_idx;
        const uint8_t* curr_mask    = pad_mask + pad_thread_offset;
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            #pragma unroll
            for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                elements_input[i][it + element] = -std::numeric_limits<float>::infinity();
            }
 
            if (element_index < batch_element_count) {
                int itr_jmp = it * WARP_SIZE;
                int itr_idx = i * element_count + itr_jmp;
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], src + itr_idx);
                apply_mask<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], 
                                                          (__half)-std::numeric_limits<float>::infinity(), 
                                                          curr_mask + itr_jmp);
            }
 
        }
    }
 
    // convert input_t to acc_t
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = elements_input[i][it];
        }
    }
 
    constexpr uint32_t  FULL_MASK = 0xffffffff;
 
    // compute local max_value
 
    // take the max_value of the first element to avoid one max call
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        max_value[i] = elements[i][0];
    }
 
    #pragma unroll
    for (int it = 1;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
 
    // reduction max_value
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float val[WARP_BATCH];
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
        }
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }
 
    // compute local sum
    acc_t sum[WARP_BATCH] { 0.0f };
 
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            //elements[i][it] = expf(elements[i][it] - max_value[i]);
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }
 
    // reduction sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }
 
    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                //dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
                output_t out[ELEMENTS_PER_LDG_STG];
                for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                    out[element] = elements[i][it + element] / sum[i];
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + i * element_count + it * WARP_SIZE, out);
            }
            else {
                break;
            }
        }
    }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using time_masked_softmax_forward_func = void(*)(input_t *dst, const output_t *src, const uint8_t *pad_mask, int batch_size, int stride, int element_count, int mod_seq_len);
 
template <typename input_t, typename output_t, typename acc_t>
bool warp_time_masked_softmax_kernel(int log2_elements, int &warp_size, int &batches_per_warp, time_masked_softmax_forward_func<input_t, output_t> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;
 
    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
 
    switch (log2_elements) {
    case 0: // 1
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,1,1>;
        break;
    case 1: // 2
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,2,1>;
        break;
    case 2: // 4
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,4,1>;
        break;
    case 3: // 8
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,8,1>;
        break;
    case 4: // 16
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,16,1>;
        break;
    case 5: // 32
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,1,32,1>;
        break;
    case 6: // 64
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,2,32,1>;
        break;
    case 7: // 128
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,4,32,1>;
        break;
    case 8: // 256
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 1,8,32,1>;
        break;
    case 9: // 512
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 1,16,32,1>;
        break;
    case 10: // 1024
        kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 1,32,32,1>;
        break;
    default:
        return false;
    }
    return true;
}
 
template<typename input_t, typename output_t, typename acc_t>
bool dispatch_time_masked_softmax(output_t *dst, const input_t *src, const uint8_t *pad_mask, int softmax_elements, int softmax_elements_stride, int batch_count, int mod_seq_len)
{
    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 1024) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;
 
        time_masked_softmax_forward_func<input_t, output_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_time_masked_softmax_kernel<input_t, output_t, acc_t>(log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }
 
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
 
        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);
 
        // compute launch size
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
 
        // launch
        kernel<<<blocks, threads>>>(dst, src, pad_mask, batch_count, softmax_elements_stride, softmax_elements, mod_seq_len);
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp softmax backward
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE=32, int ELEMENTS_PER_LDG_STG=1>
__global__ void softmax_warp_backward(__half *gradInput, const __half *grad, const __half *output, int batch_size, int stride, int element_count)
{
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
 
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;
 
    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
 
    // the first element to process by the current thread
    int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;
 
    // load data from global memory
    input_t grad_reg_input[WARP_BATCH][WARP_ITERATIONS] = {0.0f};
    input_t output_reg_input[WARP_BATCH][WARP_ITERATIONS] = {0.0f};
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&grad_reg_input[i][it], grad + i * element_count + it * WARP_SIZE);
                copy_vector<input_t,ELEMENTS_PER_LDG_STG>(&output_reg_input[i][it], output + i * element_count + it * WARP_SIZE);
            }
 
        }
    }
 
    // convert half to floating point
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            grad_reg[i][it] = grad_reg_input[i][it];
            output_reg[i][it] = output_reg_input[i][it];
        }
    }
 
 
    // compute thread local sum
    acc_t sum[WARP_BATCH] = {0};
    #pragma unroll
    for (int it = 0;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += grad_reg[i][it] * output_reg[i][it];
 
        }
    }
 
    // reduction sum
    constexpr uint32_t FULL_MASK = 0xffffffff;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }
 
    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                output_t out[ELEMENTS_PER_LDG_STG];
                for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                    out[element] = (output_reg[i][it+element] * (grad_reg[i][it+element] - sum[i]));
                }
                // store them in global memory
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(gradInput + i * element_count + it * WARP_SIZE, out);
            }
        }
    }
}
 
 
 
// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using softmax_backward_func = void(*)(output_t *gradInput, const input_t *grad, const input_t *output, int batch_size, int stride, int element_count);
 
template <typename input_t, typename output_t, typename acc_t>
bool warp_softmax_backward_kernel(int log2_elements, int &warp_size, int &batches_per_warp, softmax_backward_func<input_t, output_t> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;
 
    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
 
    switch (log2_elements) {
    case 0: // 1
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,1,1>;
        break;
    case 1: // 2
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,2,1>;
        break;
    case 2: // 4
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,4,1>;
        break;
    case 3: // 8
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,8,1>;
        break;
    case 4: // 16
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,16,1>;
        break;
    case 5: // 32
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,32,1>;
        break;
    case 6: // 64
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,2,32,1>;
        break;
    case 7: // 128
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,4,32,1>;
        break;
    case 8: // 256
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1,8,32,1>;
        break;
    case 9: // 512
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1,16,32,1>;
        break;
    case 10: // 1024
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1,32,32,1>;
        break;
    default:
        return false;
    }
    return true;
}
 
template<typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 1024) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;
 
        softmax_backward_func<input_t, output_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_softmax_backward_kernel<input_t, output_t, acc_t>(log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }
 
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
 
        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);
 
        // compute launch size
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
 
        // launch
        kernel<<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        return true;
    }
    return false;
}

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE=32, int ELEMENTS_PER_LDG_STG=1>
__global__ void masked_softmax_warp_backward(__half *gradInput, const __half *grad, const __half *output, const uint8_t *pad_mask, int batch_size, int stride, int element_count, int pad_batch_stride)
{
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
 
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;
 
    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
 
    // the first element to process by the current thread
    int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;
 
    // load data from global memory
    input_t grad_reg_input[WARP_BATCH][WARP_ITERATIONS] = {0.0f};
    input_t output_reg_input[WARP_BATCH][WARP_ITERATIONS] = {0.0f};
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&grad_reg_input[i][it], grad + i * element_count + it * WARP_SIZE);
                copy_vector<input_t,ELEMENTS_PER_LDG_STG>(&output_reg_input[i][it], output + i * element_count + it * WARP_SIZE);
            }
 
        }
    }
 
    // convert half to floating point
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            grad_reg[i][it] = grad_reg_input[i][it];
            output_reg[i][it] = output_reg_input[i][it];
        }
    }
 
 
    // compute thread local sum
    acc_t sum[WARP_BATCH] = {0};
    #pragma unroll
    for (int it = 0;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += grad_reg[i][it] * output_reg[i][it];
 
        }
    }
 
    // reduction sum
    constexpr uint32_t FULL_MASK = 0xffffffff;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }
 
    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        int pad_thread_offset = ( (first_batch + i) / pad_batch_stride) * stride + ELEMENTS_PER_LDG_STG * local_idx;
        const uint8_t* curr_mask    = pad_mask + pad_thread_offset;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                output_t out[ELEMENTS_PER_LDG_STG];
                for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                    out[element] = (output_reg[i][it+element] * (grad_reg[i][it+element] - sum[i]));
                }
                // store them in global memory
                int itr_jmp = it * WARP_SIZE;
                int itr_idx = i * element_count + itr_jmp;
                // It is kind of unfortunate this has to be here to zero something out that is close to
                // zero in the first place
                apply_mask<input_t, ELEMENTS_PER_LDG_STG>(&out[0], 0.0, curr_mask + itr_jmp);
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(gradInput + itr_idx, out);
            }
        }
    }
}
 
 
 
// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using masked_softmax_backward_func = void(*)(output_t *gradInput, const input_t *grad, const input_t *output, const uint8_t *pad_mask, int batch_size, int stride, int element_count, int pad_batch_stride);
 
template <typename input_t, typename output_t, typename acc_t>
bool warp_masked_softmax_backward_kernel(int log2_elements, int &warp_size, int &batches_per_warp, masked_softmax_backward_func<input_t, output_t> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;
 
    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
 
    switch (log2_elements) {
    case 0: // 1
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,1,1,1>;
        break;
    case 1: // 2
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,1,2,1>;
        break;
    case 2: // 4
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,1,4,1>;
        break;
    case 3: // 8
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,1,8,1>;
        break;
    case 4: // 16
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,1,16,1>;
        break;
    case 5: // 32
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,1,32,1>;
        break;
    case 6: // 64
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,2,32,1>;
        break;
    case 7: // 128
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 2,4,32,1>;
        break;
    case 8: // 256
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 1,8,32,1>;
        break;
    case 9: // 512
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 1,16,32,1>;
        break;
    case 10: // 1024
        kernel = &masked_softmax_warp_backward<input_t, output_t, acc_t, 1,32,32,1>;
        break;
    default:
        return false;
    }
    return true;
}
 
template<typename input_t, typename output_t, typename acc_t>
bool dispatch_masked_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output, const uint8_t *pad_mask, int softmax_elements, int softmax_elements_stride, int batch_count, int pad_batch_stride)
{
    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 1024) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;
 
        masked_softmax_backward_func<input_t, output_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_masked_softmax_backward_kernel<input_t, output_t, acc_t>(log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }
 
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
 
        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);
 
        // compute launch size
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
 
        // launch
        kernel<<<blocks, threads>>>(grad_input, grad, output, pad_mask, batch_count, softmax_elements_stride, softmax_elements, pad_batch_stride);
        return true;
    }
    return false;
}
