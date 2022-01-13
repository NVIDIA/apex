#pragma once
#include "philox.cuh"
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <curand_kernel.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <assert.h>
#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>
#include <limits>
#include <stdint.h>

namespace {
template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void apply_mask(Datatype *dst, Datatype value,
                                      const uint8_t *src);

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void apply_additive_mask(Datatype *dst,
                                               const Datatype *additive_mask);

template <>
__device__ __inline__ void copy_vector<__half, 1>(__half *dst,
                                                  const __half *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<float, 1>(float *dst, const float *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<__half, 4>(__half *dst,
                                                  const __half *src) {
  *((float2 *)dst) = *((float2 *)src);
}
template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst,
                                                   const uint8_t *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst,
                                                   const uint8_t *src) {
  *((half2 *)dst) = *((half2 *)src);
}

template <>
__device__ __inline__ void apply_mask<__half, 1>(__half *dst, __half value,
                                                 const uint8_t *src) {
  if (*src == 1) {
    *dst = value;
  }
}

template <>
__device__ __inline__ void
apply_additive_mask<__half, 1>(__half *dst, const __half *additive_mask) {
  *dst += *additive_mask;
}

template <>
__device__ __inline__ void
apply_additive_mask<__half, 4>(__half *dst, const __half *additive_mask) {
  *dst += *additive_mask;
  *(dst + 1) += *(additive_mask + 1);
  *(dst + 2) += *(additive_mask + 2);
  *(dst + 3) += *(additive_mask + 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp Softmax forward
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void softmax_warp_forward(input_t *dst, const output_t *src,
                                     int batch_size, int stride,
                                     int element_count) {
  assert(ELEMENTS_PER_LDG_STG == 1);

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
  dst += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;

  // load data from global memory
  input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
#pragma unroll
      for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
        elements_input[i][it + element] =
            -std::numeric_limits<float>::infinity();
      }

      if (element_index < batch_element_count) {
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(
            &elements_input[i][it], src + i * element_count + it * WARP_SIZE);
      }
    }
  }

  // convert input_t to acc_t
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = elements_input[i][it];
    }
  }

  constexpr uint32_t FULL_MASK = 0xffffffff;

  // compute local max_value

  // take the max_value of the first element to avoid one max call
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
  }

#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }

// reduction max_value
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float val[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
    }
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
    }
  }

  // compute local sum
  acc_t sum[WARP_BATCH]{0.0f};

#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      // elements[i][it] = expf(elements[i][it] - max_value[i]);
      elements[i][it] = std::exp(elements[i][it] - max_value[i]);
      sum[i] += elements[i][it];
    }
  }

// reduction sum
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        output_t out[ELEMENTS_PER_LDG_STG];
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = elements[i][it + element] / sum[i];
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count + it * WARP_SIZE, out);
      } else {
        break;
      }
    }
  }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using softmax_forward_func = void (*)(input_t *dst, const output_t *src,
                                      int batch_size, int stride,
                                      int element_count);

template <typename input_t, typename output_t, typename acc_t>
bool warp_softmax_kernel(int log2_elements, int &warp_size,
                         int &batches_per_warp,
                         softmax_forward_func<input_t, output_t> &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  switch (log2_elements) {
  case 0: // 1
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 1, 1>;
    break;
  case 1: // 2
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 2, 1>;
    break;
  case 2: // 4
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 4, 1>;
    break;
  case 3: // 8
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 8, 1>;
    break;
  case 4: // 16
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 16, 1>;
    break;
  case 5: // 32
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 32, 1>;
    break;
  case 6: // 64
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 2, 32, 1>;
    break;
  case 7: // 128
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2, 4, 32, 1>;
    break;
  case 8: // 256
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1, 8, 32, 1>;
    break;
  case 9: // 512
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1, 16, 32, 1>;
    break;
  case 10: // 1024
    kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1, 32, 32, 1>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax(output_t *dst, const input_t *src, int softmax_elements,
                      int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    softmax_forward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_softmax_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        dst, src, batch_count, softmax_elements_stride, softmax_elements);
    return true;
  }
  return false;
}

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE, int ELEMENTS_PER_LDG_STG>
__global__ void additive_masked_softmax_dropout_warp_forward_vec4(
    output_t *dst, uint8_t *dropout_mask, const input_t *src,
    const input_t *pad_mask, int batch_size, int stride, int element_count,
    int pad_batch_stride, at::PhiloxCudaState philox_args, float p) {

  assert(ELEMENTS_PER_LDG_STG == 4);
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
  int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
  acc_t pinv = acc_t(1) / p;
  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;
  // vectorize if element_count is multiple of 4, else don't vectorize
  input_t elements_input[WARP_BATCH][WARP_ITERATIONS];

  int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
  src += thread_offset;
  dst += thread_offset;
  dropout_mask += thread_offset;

  // load data from global memory
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    int pad_thread_offset = ((first_batch + i) / pad_batch_stride) * stride +
                            ELEMENTS_PER_LDG_STG * local_idx;
    const half *curr_mask = pad_mask + pad_thread_offset;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
#pragma unroll
      for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
        // masking_value is a large negative value
        elements_input[i][it + element] = -10000;
      }

      if (element_index < batch_element_count) {
        int itr_jmp = it * WARP_SIZE;
        int itr_idx = i * element_count + itr_jmp;
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it],
                                                   src + itr_idx);
        apply_additive_mask<input_t, ELEMENTS_PER_LDG_STG>(
            &elements_input[i][it],
            curr_mask +
                itr_jmp); //(__half)-std::numeric_limits<float>::infinity()
      }
    }
  }
  // convert input_t to acc_t
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = elements_input[i][it];
    }
  }

  constexpr uint32_t FULL_MASK = 0xffffffff;

  // compute local max_value

  // take the max_value of the first element to avoid one max call
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
  }

#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }

// reduction max_value
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float val[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
    }
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
    }
  }

  // compute local sum
  acc_t sum[WARP_BATCH]{0.0f};

#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = std::exp(elements[i][it] - max_value[i]);
      sum[i] += elements[i][it];
    }
  }

// reduction sum
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }
  auto seeds = at::cuda::philox::unpack(philox_args);
  Philox ph(std::get<0>(seeds), tid, std::get<1>(seeds));
  uint8_t rands[WARP_BATCH][WARP_ITERATIONS];
  float4 rand_num;
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        rand_num = uniform4(ph());
        rands[i][it] = (rand_num.x <= p) > 0.5;
        rands[i][it + 1] = (rand_num.y <= p) > 0.5;
        rands[i][it + 2] = (rand_num.z <= p) > 0.5;
        rands[i][it + 3] = (rand_num.w <= p) > 0.5;
        copy_vector<uint8_t, ELEMENTS_PER_LDG_STG>(
            dropout_mask + i * element_count + it * WARP_SIZE, &rands[i][it]);
      }
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = rands[i][it + element] *
                         (pinv * (elements[i][it + element] / sum[i]));
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count + it * WARP_SIZE, out);

      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE, int ELEMENTS_PER_LDG_STG>
__global__ void additive_masked_softmax_dropout_warp_forward(
    output_t *dst, uint8_t *dropout_mask, const input_t *src,
    const input_t *pad_mask, int batch_size, int stride, int element_count,
    int pad_batch_stride, at::PhiloxCudaState philox_args, float p) {
  assert(ELEMENTS_PER_LDG_STG == 1);
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
  int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
  acc_t pinv = acc_t(1) / p;
  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;
  // vectorize if element_count is multiple of 4, else don't vectorize
  input_t elements_input[WARP_BATCH][WARP_ITERATIONS];

  int thread_offset = first_batch * stride + local_idx;
  src += thread_offset;
  dst += thread_offset;
  dropout_mask += thread_offset;

  // load data from global memory
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    int pad_thread_offset =
        ((first_batch + i) / pad_batch_stride) * stride + local_idx;
    const half *curr_mask = pad_mask + pad_thread_offset;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += 1) {
      int element_index = local_idx + it * WARP_SIZE;
#pragma unroll
      for (int element = 0; element < 1; ++element) {
        // masking_value is a large negative value
        elements_input[i][it + element] = -10000;
      }

      if (element_index < batch_element_count) {
        int itr_jmp = it * WARP_SIZE;
        int itr_idx = i * element_count + itr_jmp;
        copy_vector<input_t, 1>(&elements_input[i][it], src + itr_idx);
        apply_additive_mask<input_t, 1>(&elements_input[i][it],
                                        curr_mask + itr_jmp);
      }
    }
  }
  // convert input_t to acc_t
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = elements_input[i][it];
    }
  }

  constexpr uint32_t FULL_MASK = 0xffffffff;

  // compute local max_value

  // take the max_value of the first element to avoid one max call
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
  }

#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }

// reduction max_value
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float val[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
    }
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
    }
  }

  // compute local sum
  acc_t sum[WARP_BATCH]{0.0f};

#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = std::exp(elements[i][it] - max_value[i]);
      sum[i] += elements[i][it];
    }
  }

// reduction sum
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }
  curandStatePhilox4_32_10_t state;
  auto seeds = at::cuda::philox::unpack(philox_args);
  curand_init(std::get<0>(seeds), tid, std::get<1>(seeds), &state);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += 1) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        output_t out[1];
        acc_t softmax_out[1];
        uint8_t dropout_mask_temp[1];
        // generate a vector of random numbers here
        float rand = curand_uniform(&state);
        float *rand_ptr = (float *)(&rand);
#pragma unroll
        for (int element = 0; element < 1; ++element) {
          softmax_out[element] = (elements[i][it + element] / sum[i]);
          rand_ptr[element] = rand_ptr[element] <= p;
          out[element] = rand_ptr[element] * pinv * softmax_out[element];
          dropout_mask_temp[element] =
              rand_ptr[element] > 0.5; // just to distinguish 0.0f and 1.0f
        }
        copy_vector<output_t, 1>(dst + i * element_count + it * WARP_SIZE, out);
        copy_vector<uint8_t, 1>(dropout_mask + i * element_count +
                                    it * WARP_SIZE,
                                dropout_mask_temp);

      } else {
        break;
      }
    }
  }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t>
using additive_masked_softmax_dropout_forward_func = void (*)(
    output_t *dst, uint8_t *dropout_mask, const input_t *src,
    const input_t *pad_mask, int batch_size, int stride, int element_count,
    int pad_batch_stride, at::PhiloxCudaState philox_args, float p);

template <typename input_t, typename output_t, typename acc_t>
bool warp_additive_masked_softmax_dropout_kernel(
    int element_count, int log2_elements, int &warp_size, int &batches_per_warp,
    additive_masked_softmax_dropout_forward_func<input_t, output_t, acc_t>
        &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
  bool flag_vec4 = (element_count % 4 == 0);
  switch (log2_elements) {
  case 0: // 1
    kernel = &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                           acc_t, 2, 1, 1, 1>;
    break;
  case 1: // 2
    kernel = &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                           acc_t, 2, 1, 2, 1>;
    break;
  case 2: // 4
    kernel = &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                           acc_t, 2, 1, 4, 1>;
    break;
  case 3: // 8
    kernel = &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                           acc_t, 2, 1, 8, 1>;
    break;
  case 4: // 16
    kernel = &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                           acc_t, 2, 1, 16, 1>;
    break;
  case 5: // 32
    kernel = &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                           acc_t, 2, 1, 32, 1>;
    break;
  case 6: // 64
    kernel = &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                           acc_t, 2, 2, 32, 1>;
    break;
  case 7: // 128
    if (flag_vec4)
      kernel = &additive_masked_softmax_dropout_warp_forward_vec4<
          input_t, output_t, acc_t, 2, 4, 32, 4>;
    else
      kernel =
          &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                        acc_t, 2, 4, 32, 1>;
    break;
  case 8: // 256
    if (flag_vec4)
      kernel = &additive_masked_softmax_dropout_warp_forward_vec4<
          input_t, output_t, acc_t, 1, 8, 32, 4>;
    else
      kernel =
          &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                        acc_t, 1, 8, 32, 1>;
    break;
  case 9: // 512
    if (flag_vec4)
      kernel = &additive_masked_softmax_dropout_warp_forward_vec4<
          input_t, output_t, acc_t, 1, 16, 32, 4>;
    else
      kernel =
          &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                        acc_t, 1, 16, 32, 1>;
    break;
  case 10: // 1024
    if (flag_vec4)
      kernel = &additive_masked_softmax_dropout_warp_forward_vec4<
          input_t, output_t, acc_t, 1, 32, 32, 4>;
    else
      kernel =
          &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                        acc_t, 1, 32, 32, 1>;
    break;
  case 11: // 2048
    if (flag_vec4)
      kernel = &additive_masked_softmax_dropout_warp_forward_vec4<
          input_t, output_t, acc_t, 1, 64, 32, 4>;
    else
      kernel =
          &additive_masked_softmax_dropout_warp_forward<input_t, output_t,
                                                        acc_t, 1, 64, 32, 1>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_additive_masked_softmax_dropout(
    output_t *dst, uint8_t *dropout_mask, const input_t *src,
    const input_t *pad_mask, int totalElements, int softmax_elements,
    int softmax_elements_stride, int batch_count, int pad_batch_stride, float p,
    cudaStream_t streamid) // p is the probability to keep, not drop
{

  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 2048) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    additive_masked_softmax_dropout_forward_func<input_t, output_t, acc_t>
        kernel;
    int warp_size, batches_per_warp;
    if (!warp_additive_masked_softmax_dropout_kernel<input_t, output_t, acc_t>(
            softmax_elements, log2_elements, warp_size, batches_per_warp,
            kernel)) {
      return false;
    }

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    // compute warps per block.
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    c10::optional<at::Generator> gen_;
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    int64_t counter_offset = (totalElements / (blocks * threads_per_block) + 1);
    at::PhiloxCudaState rng_engine_inputs;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(counter_offset);
    }

    // compute launch size
    dim3 threads(warp_size, warps_per_block, 1);

    // launch
    kernel<<<blocks, threads, 0, streamid>>>(
        dst, dropout_mask, src, pad_mask, batch_count, softmax_elements_stride,
        softmax_elements, pad_batch_stride, rng_engine_inputs, p);
    return true;
  }
  return false;
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void additive_masked_softmax_warp_forward(
    input_t *dst, const output_t *src, const input_t *pad_mask, int batch_size,
    int stride, int element_count, int pad_batch_stride) {
  assert(ELEMENTS_PER_LDG_STG == 1);

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
  src += thread_offset;
  dst += thread_offset;

  // load data from global memory
  input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    int pad_thread_offset = ((first_batch + i) / pad_batch_stride) * stride +
                            ELEMENTS_PER_LDG_STG * local_idx;
    const half *curr_mask = pad_mask + pad_thread_offset;
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
#pragma unroll
      for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
        // masking_value is a large negative value
        elements_input[i][it + element] = -10000;
      }

      if (element_index < batch_element_count) {
        int itr_jmp = it * WARP_SIZE;
        int itr_idx = i * element_count + itr_jmp;
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it],
                                                   src + itr_idx);
        // apply_mask<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it],
        //                                          (__half)-std::numeric_limits<float>::infinity(),
        //                                          curr_mask + itr_jmp);
        elements_input[i][it] += *(curr_mask + itr_jmp);
      }
    }
  }

  // convert input_t to acc_t
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = elements_input[i][it];
    }
  }

  constexpr uint32_t FULL_MASK = 0xffffffff;

  // compute local max_value

  // take the max_value of the first element to avoid one max call
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
  }

#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }

// reduction max_value
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float val[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
    }
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
    }
  }

  // compute local sum
  acc_t sum[WARP_BATCH]{0.0f};

#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      // elements[i][it] = expf(elements[i][it] - max_value[i]);
      elements[i][it] = std::exp(elements[i][it] - max_value[i]);
      sum[i] += elements[i][it];
    }
  }

// reduction sum
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        output_t out[ELEMENTS_PER_LDG_STG];
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = elements[i][it + element] / sum[i];
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count + it * WARP_SIZE, out);
      } else {
        break;
      }
    }
  }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using additive_masked_softmax_forward_func = void (*)(
    input_t *dst, const output_t *src, const half *pad_mask, int batch_size,
    int stride, int element_count, int pad_batch_stride);

template <typename input_t, typename output_t, typename acc_t>
bool warp_additive_masked_softmax_kernel(
    int log2_elements, int &warp_size, int &batches_per_warp,
    additive_masked_softmax_forward_func<input_t, output_t> &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  switch (log2_elements) {
  case 0: // 1
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   1, 1, 1>;
    break;
  case 1: // 2
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   1, 2, 1>;
    break;
  case 2: // 4
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   1, 4, 1>;
    break;
  case 3: // 8
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   1, 8, 1>;
    break;
  case 4: // 16
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   1, 16, 1>;
    break;
  case 5: // 32
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   1, 32, 1>;
    break;
  case 6: // 64
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   2, 32, 1>;
    break;
  case 7: // 128
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 2,
                                                   4, 32, 1>;
    break;
  case 8: // 256
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 1,
                                                   8, 32, 1>;
    break;
  case 9: // 512
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 1,
                                                   16, 32, 1>;
    break;
  case 10: // 1024
    kernel = &additive_masked_softmax_warp_forward<input_t, output_t, acc_t, 1,
                                                   32, 32, 1>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_additive_masked_softmax(output_t *dst, const input_t *src,
                                      const input_t *pad_mask,
                                      int softmax_elements,
                                      int softmax_elements_stride,
                                      int batch_count, int pad_batch_stride) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    additive_masked_softmax_forward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_additive_masked_softmax_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        dst, src, pad_mask, batch_count, softmax_elements_stride,
        softmax_elements, pad_batch_stride);
    return true;
  }
  return false;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_additive_masked_softmax_stream(
    output_t *dst, const input_t *src, const input_t *pad_mask,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    int pad_batch_stride, cudaStream_t streamid) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;
    additive_masked_softmax_forward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_additive_masked_softmax_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, streamid>>>(
        dst, src, pad_mask, batch_count, softmax_elements_stride,
        softmax_elements, pad_batch_stride);
    return true;
  }
  return false;
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void
masked_softmax_warp_forward(input_t *dst, const output_t *src,
                            const uint8_t *pad_mask, int batch_size, int stride,
                            int element_count, int pad_batch_stride) {
  assert(ELEMENTS_PER_LDG_STG == 1);

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
  src += thread_offset;
  dst += thread_offset;

  // load data from global memory
  input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    int pad_thread_offset = ((first_batch + i) / pad_batch_stride) * stride +
                            ELEMENTS_PER_LDG_STG * local_idx;
    const uint8_t *curr_mask = pad_mask + pad_thread_offset;
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
#pragma unroll
      for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
        elements_input[i][it + element] =
            -std::numeric_limits<float>::infinity();
      }

      if (element_index < batch_element_count) {
        int itr_jmp = it * WARP_SIZE;
        int itr_idx = i * element_count + itr_jmp;
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it],
                                                   src + itr_idx);
        apply_mask<input_t, ELEMENTS_PER_LDG_STG>(
            &elements_input[i][it],
            (__half)-std::numeric_limits<float>::infinity(),
            curr_mask + itr_jmp);
      }
    }
  }

  // convert input_t to acc_t
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = elements_input[i][it];
    }
  }

  constexpr uint32_t FULL_MASK = 0xffffffff;

  // compute local max_value

  // take the max_value of the first element to avoid one max call
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
  }

#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }

// reduction max_value
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float val[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
    }
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
    }
  }

  // compute local sum
  acc_t sum[WARP_BATCH]{0.0f};

#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      // elements[i][it] = expf(elements[i][it] - max_value[i]);
      elements[i][it] = std::exp(elements[i][it] - max_value[i]);
      sum[i] += elements[i][it];
    }
  }

// reduction sum
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        output_t out[ELEMENTS_PER_LDG_STG];
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = elements[i][it + element] / sum[i];
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count + it * WARP_SIZE, out);
      } else {
        break;
      }
    }
  }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using masked_softmax_forward_func = void (*)(input_t *dst, const output_t *src,
                                             const uint8_t *pad_mask,
                                             int batch_size, int stride,
                                             int element_count,
                                             int pad_batch_stride);

template <typename input_t, typename output_t, typename acc_t>
bool warp_masked_softmax_kernel(
    int log2_elements, int &warp_size, int &batches_per_warp,
    masked_softmax_forward_func<input_t, output_t> &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  switch (log2_elements) {
  case 0: // 1
    kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 1, 1>;
    break;
  case 1: // 2
    kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 2, 1>;
    break;
  case 2: // 4
    kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 4, 1>;
    break;
  case 3: // 8
    kernel = &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 8, 1>;
    break;
  case 4: // 16
    kernel =
        &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 16, 1>;
    break;
  case 5: // 32
    kernel =
        &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 32, 1>;
    break;
  case 6: // 64
    kernel =
        &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 2, 32, 1>;
    break;
  case 7: // 128
    kernel =
        &masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 4, 32, 1>;
    break;
  case 8: // 256
    kernel =
        &masked_softmax_warp_forward<input_t, output_t, acc_t, 1, 8, 32, 1>;
    break;
  case 9: // 512
    kernel =
        &masked_softmax_warp_forward<input_t, output_t, acc_t, 1, 16, 32, 1>;
    break;
  case 10: // 1024
    kernel =
        &masked_softmax_warp_forward<input_t, output_t, acc_t, 1, 32, 32, 1>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_masked_softmax(output_t *dst, const input_t *src,
                             const uint8_t *pad_mask, int softmax_elements,
                             int softmax_elements_stride, int batch_count,
                             int pad_batch_stride) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    masked_softmax_forward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_masked_softmax_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        dst, src, pad_mask, batch_count, softmax_elements_stride,
        softmax_elements, pad_batch_stride);
    return true;
  }
  return false;
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void time_masked_softmax_warp_forward(
    input_t *dst, const output_t *src, const uint8_t *pad_mask, int batch_size,
    int stride, int element_count, int mod_seq_len) {
  assert(ELEMENTS_PER_LDG_STG == 1);

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
  src += thread_offset;
  dst += thread_offset;

  // load data from global memory
  input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    int pad_thread_offset = ((first_batch + i) % mod_seq_len) * stride +
                            ELEMENTS_PER_LDG_STG * local_idx;
    const uint8_t *curr_mask = pad_mask + pad_thread_offset;
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
#pragma unroll
      for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
        elements_input[i][it + element] =
            -std::numeric_limits<float>::infinity();
      }

      if (element_index < batch_element_count) {
        int itr_jmp = it * WARP_SIZE;
        int itr_idx = i * element_count + itr_jmp;
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it],
                                                   src + itr_idx);
        apply_mask<input_t, ELEMENTS_PER_LDG_STG>(
            &elements_input[i][it],
            (__half)-std::numeric_limits<float>::infinity(),
            curr_mask + itr_jmp);
      }
    }
  }

  // convert input_t to acc_t
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = elements_input[i][it];
    }
  }

  constexpr uint32_t FULL_MASK = 0xffffffff;

  // compute local max_value

  // take the max_value of the first element to avoid one max call
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
  }

#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }

// reduction max_value
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float val[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
    }
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
    }
  }

  // compute local sum
  acc_t sum[WARP_BATCH]{0.0f};

#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      // elements[i][it] = expf(elements[i][it] - max_value[i]);
      elements[i][it] = std::exp(elements[i][it] - max_value[i]);
      sum[i] += elements[i][it];
    }
  }

// reduction sum
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        output_t out[ELEMENTS_PER_LDG_STG];
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = elements[i][it + element] / sum[i];
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count + it * WARP_SIZE, out);
      } else {
        break;
      }
    }
  }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using time_masked_softmax_forward_func =
    void (*)(input_t *dst, const output_t *src, const uint8_t *pad_mask,
             int batch_size, int stride, int element_count, int mod_seq_len);

template <typename input_t, typename output_t, typename acc_t>
bool warp_time_masked_softmax_kernel(
    int log2_elements, int &warp_size, int &batches_per_warp,
    time_masked_softmax_forward_func<input_t, output_t> &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  switch (log2_elements) {
  case 0: // 1
    kernel =
        &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 1, 1>;
    break;
  case 1: // 2
    kernel =
        &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 2, 1>;
    break;
  case 2: // 4
    kernel =
        &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 4, 1>;
    break;
  case 3: // 8
    kernel =
        &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1, 8, 1>;
    break;
  case 4: // 16
    kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1,
                                               16, 1>;
    break;
  case 5: // 32
    kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 1,
                                               32, 1>;
    break;
  case 6: // 64
    kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 2,
                                               32, 1>;
    break;
  case 7: // 128
    kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 2, 4,
                                               32, 1>;
    break;
  case 8: // 256
    kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 1, 8,
                                               32, 1>;
    break;
  case 9: // 512
    kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 1, 16,
                                               32, 1>;
    break;
  case 10: // 1024
    kernel = &time_masked_softmax_warp_forward<input_t, output_t, acc_t, 1, 32,
                                               32, 1>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_time_masked_softmax(output_t *dst, const input_t *src,
                                  const uint8_t *pad_mask, int softmax_elements,
                                  int softmax_elements_stride, int batch_count,
                                  int mod_seq_len) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    time_masked_softmax_forward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_time_masked_softmax_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        dst, src, pad_mask, batch_count, softmax_elements_stride,
        softmax_elements, mod_seq_len);
    return true;
  }
  return false;
}

int log2_ceil_native(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value)
    ++log2_value;
  return log2_value;
}

template <typename T>
__device__ __forceinline__ T
WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize,
                     unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE>
__device__ __forceinline__ void warp_reduce_sum(acc_t *sum) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
      sum[i] = sum[i] + b;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp softmax backward functions as fused variants of
// at::softmax_backward_data function
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// softmax backward data function is taken from native pytorch, elementwise mul
// is fused in the epolog, as well as masking and scaling for fusing dropout

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements, bool is_log_softmax>
__global__ void masked_scale_softmax_warp_backward_masked_dgrad(
    output_t *gradInput, const input_t *grad, const input_t *output,
    const uint8_t *mask, const uint8_t *pad_mask, acc_t scale, int batch_size,
    int stride, int element_count, int heads) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
  // warp_size of method warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x % WARP_SIZE;

  // the first element to process by the current thread
  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;
  mask += thread_offset;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified
  // to one loop, but I think doing so would obfuscate the logic of the
  // algorithm, thus I chose to keep the nested loops. This should have no
  // impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        grad_reg[i][it] =
            (input_t)((acc_t)mask[i * element_count + it * WARP_SIZE] *
                      (acc_t)grad[i * element_count + it * WARP_SIZE] *
                      (acc_t)scale) *
            output[i * element_count + it * WARP_SIZE];
        output_reg[i][it] = output[i * element_count + it * WARP_SIZE];
      } else {
        grad_reg[i][it] = acc_t(0);
        output_reg[i][it] = acc_t(0);
      }
    }
  }

  acc_t sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce_sum<acc_t, WARP_BATCH, WARP_SIZE>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        int total_ind = thread_offset + i * element_count + it * WARP_SIZE;
        int pad_mask_ind =
            element_count *
                (total_ind / (heads * element_count * element_count)) +
            total_ind % element_count;
        uint8_t pad_mask_element = 1 - pad_mask[pad_mask_ind];
        if (pad_mask_element == 0)
          gradInput[i * element_count + it * WARP_SIZE] = 0;
        else {
          if (is_log_softmax) {
            gradInput[i * element_count + it * WARP_SIZE] =
                (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
          } else {
            gradInput[i * element_count + it * WARP_SIZE] =
                (grad_reg[i][it] - output_reg[i][it] * sum[i]);
          }
        }
      }
    }
  }
}
template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_masked_scale_softmax_backward_masked_out(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const uint8_t *mask, const uint8_t *pad_mask, acc_t scale,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    int heads) {
  TORCH_INTERNAL_ASSERT(softmax_elements >= 0 && softmax_elements <= 1024);
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil_native(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside
    // softmax_warp_backward.
    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside
    // softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
    case 0: // 1
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      0, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 1: // 2
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      1, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 2: // 4
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      2, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 3: // 8
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      3, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 4: // 16
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      4, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 5: // 32
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      5, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 6: // 64
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      6, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 7: // 128
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      7, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 8: // 256
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      8, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 9: // 512
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      9, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 10: // 1024
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      10, is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    default:
      break;
    }
  }
}

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_masked_scale_softmax_backward_masked_out_stream(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const uint8_t *mask, const uint8_t *pad_mask, acc_t scale,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    int heads, cudaStream_t streamid) {
  TORCH_INTERNAL_ASSERT(softmax_elements >= 0 && softmax_elements <= 1024);
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil_native(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;
    // This value must match the WARP_SIZE constexpr value computed inside
    // softmax_warp_backward.
    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    // This value must match the WARP_BATCH constexpr value computed inside
    // softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
    case 0: // 1
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      0, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 1: // 2
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      1, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 2: // 4
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      2, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 3: // 8
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      3, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 4: // 16
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      4, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 5: // 32
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      5, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 6: // 64
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      6, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 7: // 128
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      7, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 8: // 256
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      8, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 9: // 512
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      9, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    case 10: // 1024
      masked_scale_softmax_warp_backward_masked_dgrad<input_t, output_t, acc_t,
                                                      10, is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, pad_mask, scale, batch_count,
              softmax_elements_stride, softmax_elements, heads);
      break;
    default:
      break;
    }
  }
}

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements, bool is_log_softmax>
__global__ void
masked_scale_softmax_warp_backward(output_t *gradInput, const input_t *grad,
                                   const input_t *output, const uint8_t *mask,
                                   acc_t scale, int batch_size, int stride,
                                   int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
  // warp_size of method warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x % WARP_SIZE;

  // the first element to process by the current thread
  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;
  mask += thread_offset;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified
  // to one loop, but I think doing so would obfuscate the logic of the
  // algorithm, thus I chose to keep the nested loops. This should have no
  // impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        grad_reg[i][it] =
            (input_t)((acc_t)mask[i * element_count + it * WARP_SIZE] *
                      (acc_t)grad[i * element_count + it * WARP_SIZE] *
                      (acc_t)scale) *
            output[i * element_count + it * WARP_SIZE];
        output_reg[i][it] = output[i * element_count + it * WARP_SIZE];
      } else {
        grad_reg[i][it] = acc_t(0);
        output_reg[i][it] = acc_t(0);
      }
    }
  }

  acc_t sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce_sum<acc_t, WARP_BATCH, WARP_SIZE>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        if (is_log_softmax) {
          gradInput[i * element_count + it * WARP_SIZE] =
              (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
        } else {
          gradInput[i * element_count + it * WARP_SIZE] =
              (grad_reg[i][it] - output_reg[i][it] * sum[i]);
        }
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG,
          bool is_log_softmax>
__global__ void masked_scale_softmax_warp_backward_recompute(
    output_t *gradInput, const input_t *grad, const input_t *softmax_input,
    const input_t *pad_mask, const uint8_t *mask, acc_t scale, int batch_size,
    int stride, int pad_batch_stride, int element_count) {
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x % WARP_SIZE;
  // vectorize if a row length is multiple of 4
  int flag_vec4 = element_count & 3 == 0;
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  input_t elements_input[WARP_BATCH][WARP_ITERATIONS];

  // the first element to process by the current thread
  int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;

  grad += thread_offset;
  softmax_input += thread_offset;
  gradInput += thread_offset;
  mask += thread_offset;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified
  // to one loop, but I think doing so would obfuscate the logic of the
  // algorithm, thus I chose to keep the nested loops. This should have no
  // impact on performance because the loops are unrolled anyway.

  // load data from global memory
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    int pad_thread_offset = ((first_batch + i) / pad_batch_stride) * stride +
                            ELEMENTS_PER_LDG_STG * local_idx;
    const input_t *curr_mask = pad_mask + pad_thread_offset;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

#pragma unroll
      for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
        // masking_value is a large negative value
        elements_input[i][it + element] = -10000;
        grad_reg[i][it + element] = acc_t(0);
      }

      if (element_index < batch_element_count) {
        int itr_jmp = it * WARP_SIZE;
        int itr_idx = i * element_count + itr_jmp;
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it],
                                                   softmax_input + itr_idx);
        apply_additive_mask<input_t, ELEMENTS_PER_LDG_STG>(
            &elements_input[i][it],
            curr_mask +
                itr_jmp); //(__half)-std::numeric_limits<float>::infinity()
        uint8_t mask_temp[ELEMENTS_PER_LDG_STG];
        input_t grad_temp[ELEMENTS_PER_LDG_STG];
        copy_vector<uint8_t, ELEMENTS_PER_LDG_STG>(&mask_temp[0],
                                                   mask + itr_idx);
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&grad_temp[0],
                                                   grad + itr_idx);
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          grad_reg[i][it + element] =
              ((acc_t)mask_temp[element] * (acc_t)grad_temp[element] *
               (acc_t)scale);
        }
      }
    }
  }
  // load data from global memory

  // convert input_t to acc_t
  // TODO : remove this, input is already acc_t type in register
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = elements_input[i][it];
    }
  }

  constexpr uint32_t FULL_MASK = 0xffffffff;

  // compute local max_value

  // take the max_value of the first element to avoid one max call
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
  }

#pragma unroll
  for (int it = 1; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }

// reduction max_value
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float val[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
    }
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
    }
  }

  // compute local sum
  acc_t sum[WARP_BATCH]{0.0f};

#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      // elements[i][it] = expf(elements[i][it] - max_value[i]);
      elements[i][it] = std::exp(elements[i][it] - max_value[i]);
      sum[i] += elements[i][it];
    }
  }

// reduction sum
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it++) {
      elements[i][it] = elements[i][it] / sum[i];
      grad_reg[i][it] = grad_reg[i][it] * elements[i][it];
    }
  }

  acc_t grad_sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    grad_sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      grad_sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce_sum<acc_t, WARP_BATCH, WARP_SIZE>(grad_sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        output_t grad_input_reg[ELEMENTS_PER_LDG_STG];
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; element++) {
          if (is_log_softmax) {
            grad_input_reg[element] =
                (grad_reg[i][it + element] -
                 std::exp(elements[i][it + element]) * grad_sum[i]);
          } else {
            grad_input_reg[element] = (grad_reg[i][it + element] -
                                       elements[i][it + element] * grad_sum[i]);
          }
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            gradInput + i * element_count + it * WARP_SIZE, grad_input_reg);
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
using masked_scale_softmax_warp_backward_recompute_func = void (*)(
    output_t *gradInput, const input_t *grad, const input_t *softmax_input,
    const input_t *pad_mask, const uint8_t *mask, acc_t scale, int batch_size,
    int stride, int pad_batch_stride, int element_count);

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
bool masked_scale_softmax_warp_backward_recompute_kernel(
    int element_count, int log2_elements, int &warp_size, int &batches_per_warp,
    masked_scale_softmax_warp_backward_recompute_func<input_t, output_t, acc_t,
                                                      is_log_softmax> &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
  bool flag_vec4 = (element_count % 4 == 0);
  switch (log2_elements) {
  case 0: // 1
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 1, 1, 1, is_log_softmax>;
    break;
  case 1: // 2
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 1, 2, 1, is_log_softmax>;
    break;
  case 2: // 4
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 1, 4, 1, is_log_softmax>;
    break;
  case 3: // 8
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 1, 8, 1, is_log_softmax>;
    break;
  case 4: // 16
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 1, 16, 1, is_log_softmax>;
    break;
  case 5: // 32
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 1, 32, 1, is_log_softmax>;
    break;
  case 6: // 64
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 2, 32, 1, is_log_softmax>;
    break;
  case 7: // 128
    kernel = &masked_scale_softmax_warp_backward_recompute<
        input_t, output_t, acc_t, 2, 4, 32, 1, is_log_softmax>;
    break;
  case 8: // 256
    if (flag_vec4)
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 8, 32, 4, is_log_softmax>;
    else
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 8, 32, 1, is_log_softmax>;
    break;
  case 9: // 512
    if (flag_vec4)
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 16, 32, 4, is_log_softmax>;
    else
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 16, 32, 1, is_log_softmax>;
    break;
  case 10: // 1024
    if (flag_vec4)
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 32, 32, 4, is_log_softmax>;
    else
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 32, 32, 1, is_log_softmax>;
    break;
  case 11: // 2048
    if (flag_vec4)
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 64, 32, 4, is_log_softmax>;
    else
      kernel = &masked_scale_softmax_warp_backward_recompute<
          input_t, output_t, acc_t, 1, 64, 32, 1, is_log_softmax>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
bool dispatch_masked_scale_softmax_backward_recompute(
    output_t *grad_input, const input_t *grad, const input_t *softmax_input,
    const input_t *pad_mask, const uint8_t *mask, acc_t scale,
    int softmax_elements, int softmax_elements_stride, int pad_batch_stride,
    int batch_count, cudaStream_t streamid) {

  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 2048) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    masked_scale_softmax_warp_backward_recompute_func<input_t, output_t, acc_t,
                                                      is_log_softmax>
        kernel;
    int warp_size, batches_per_warp;
    if (!masked_scale_softmax_warp_backward_recompute_kernel<
            input_t, output_t, acc_t, is_log_softmax>(
            softmax_elements, log2_elements, warp_size, batches_per_warp,
            kernel)) {
      return false;
    }

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    // compute warps per block.
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;

    // compute launch size
    dim3 threads(warp_size, warps_per_block, 1);

    // launch
    kernel<<<blocks, threads, 0, streamid>>>(
        grad_input, grad, softmax_input, pad_mask, mask, scale, batch_count,
        softmax_elements_stride, pad_batch_stride, softmax_elements);
    return true;
  }
  return false;
}

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_masked_scale_softmax_backward_stream(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const uint8_t *mask, acc_t scale, int softmax_elements,
    int softmax_elements_stride, int batch_count, cudaStream_t streamid) {
  TORCH_INTERNAL_ASSERT(softmax_elements >= 0 && softmax_elements <= 1024);
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil_native(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;
    // This value must match the WARP_SIZE constexpr value computed inside
    // softmax_warp_backward.
    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    // This value must match the WARP_BATCH constexpr value computed inside
    // softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
    case 0: // 1
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 0,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 1: // 2
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 1,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 2: // 4
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 2,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 3: // 8
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 3,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 4: // 16
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 4,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 5: // 32
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 5,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 6: // 64
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 6,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 7: // 128
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 7,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 8: // 256
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 8,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 9: // 512
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 9,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    case 10: // 1024
      masked_scale_softmax_warp_backward<input_t, output_t, acc_t, 10,
                                         is_log_softmax>
          <<<blocks, threads, 0, streamid>>>(
              grad_input, grad, output, mask, scale, batch_count,
              softmax_elements_stride, softmax_elements);
      break;
    default:
      break;
    }
  }
}

// elementwise multiplication called in at::softmax_backward_data is fused
// inside softmax dgrad kernel as a result of fusion, intermediate
// multiplication result is stored in fp32 in registers, instead of fp16
template <typename input_t, typename output_t, typename acc_t,
          int log2_elements, bool is_log_softmax>
__global__ void
softmax_warp_backward_fused_native(output_t *gradInput, const input_t *grad,
                                   const input_t *output, int batch_size,
                                   int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
  // warp_size of method warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x % WARP_SIZE;

  // the first element to process by the current thread
  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified
  // to one loop, but I think doing so would obfuscate the logic of the
  // algorithm, thus I chose to keep the nested loops. This should have no
  // impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        grad_reg[i][it] = grad[i * element_count + it * WARP_SIZE] *
                          output[i * element_count + it * WARP_SIZE];
        output_reg[i][it] = output[i * element_count + it * WARP_SIZE];
      } else {
        grad_reg[i][it] = acc_t(0);
        output_reg[i][it] = acc_t(0);
      }
    }
  }

  acc_t sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0]; //* output_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it]; // * output_reg[i][it];
    }
  }
  warp_reduce_sum<acc_t, WARP_BATCH, WARP_SIZE>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        if (is_log_softmax) {
          gradInput[i * element_count + it * WARP_SIZE] =
              (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
        } else {
          gradInput[i * element_count + it * WARP_SIZE] =
              (grad_reg[i][it] - output_reg[i][it] * sum[i]);
        }
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_softmax_backward_fused_native(
    output_t *grad_input, const input_t *grad, const input_t *output,
    int softmax_elements, int softmax_elements_stride, int batch_count) {
  TORCH_INTERNAL_ASSERT(softmax_elements >= 0 && softmax_elements <= 1024);
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil_native(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside
    // softmax_warp_backward.
    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside
    // softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
    case 0: // 1
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 0,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 1: // 2
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 1,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 2: // 4
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 2,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 3: // 8
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 3,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 4: // 16
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 4,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 5: // 32
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 5,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 6: // 64
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 6,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 7: // 128
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 7,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 8: // 256
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 8,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 9: // 512
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 9,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    case 10: // 1024
      softmax_warp_backward_fused_native<input_t, output_t, acc_t, 10,
                                         is_log_softmax>
          <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input, grad, output, batch_count, softmax_elements_stride,
              softmax_elements);
      break;
    default:
      break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp softmax backward
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void softmax_warp_backward(__half *gradInput, const __half *grad,
                                      const __half *output, int batch_size,
                                      int stride, int element_count) {
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
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
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(
            &grad_reg_input[i][it], grad + i * element_count + it * WARP_SIZE);
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&output_reg_input[i][it],
                                                   output + i * element_count +
                                                       it * WARP_SIZE);
      }
    }
  }

  // convert half to floating point
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      grad_reg[i][it] = grad_reg_input[i][it];
      output_reg[i][it] = output_reg_input[i][it];
    }
  }

  // compute thread local sum
  acc_t sum[WARP_BATCH] = {0};
#pragma unroll
  for (int it = 0; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += grad_reg[i][it] * output_reg[i][it];
    }
  }

  // reduction sum
  constexpr uint32_t FULL_MASK = 0xffffffff;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        output_t out[ELEMENTS_PER_LDG_STG];
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = (output_reg[i][it + element] *
                          (grad_reg[i][it + element] - sum[i]));
        }
        // store them in global memory
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            gradInput + i * element_count + it * WARP_SIZE, out);
      }
    }
  }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using softmax_backward_func = void (*)(output_t *gradInput, const input_t *grad,
                                       const input_t *output, int batch_size,
                                       int stride, int element_count);

template <typename input_t, typename output_t, typename acc_t>
bool warp_softmax_backward_kernel(
    int log2_elements, int &warp_size, int &batches_per_warp,
    softmax_backward_func<input_t, output_t> &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  switch (log2_elements) {
  case 0: // 1
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 1, 1>;
    break;
  case 1: // 2
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 2, 1>;
    break;
  case 2: // 4
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 4, 1>;
    break;
  case 3: // 8
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 8, 1>;
    break;
  case 4: // 16
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 16, 1>;
    break;
  case 5: // 32
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 32, 1>;
    break;
  case 6: // 64
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 2, 32, 1>;
    break;
  case 7: // 128
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2, 4, 32, 1>;
    break;
  case 8: // 256
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1, 8, 32, 1>;
    break;
  case 9: // 512
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1, 16, 32, 1>;
    break;
  case 10: // 1024
    kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1, 32, 32, 1>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax_backward(output_t *grad_input, const input_t *grad,
                               const input_t *output, int softmax_elements,
                               int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    softmax_backward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_softmax_backward_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_input, grad, output, batch_count, softmax_elements_stride,
        softmax_elements);
    return true;
  }
  return false;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax_backward_stream(output_t *grad_input, const input_t *grad,
                                      const input_t *output,
                                      int softmax_elements,
                                      int softmax_elements_stride,
                                      int batch_count, cudaStream_t streamid) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;
    softmax_backward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_softmax_backward_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, streamid>>>(
        grad_input, grad, output, batch_count, softmax_elements_stride,
        softmax_elements);
    return true;
  }
  return false;
}

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void
masked_softmax_warp_backward(__half *gradInput, const __half *grad,
                             const __half *output, const uint8_t *pad_mask,
                             int batch_size, int stride, int element_count,
                             int pad_batch_stride) {
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
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
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(
            &grad_reg_input[i][it], grad + i * element_count + it * WARP_SIZE);
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&output_reg_input[i][it],
                                                   output + i * element_count +
                                                       it * WARP_SIZE);
      }
    }
  }

  // convert half to floating point
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      grad_reg[i][it] = grad_reg_input[i][it];
      output_reg[i][it] = output_reg_input[i][it];
    }
  }

  // compute thread local sum
  acc_t sum[WARP_BATCH] = {0};
#pragma unroll
  for (int it = 0; it < WARP_ITERATIONS; ++it) {
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += grad_reg[i][it] * output_reg[i][it];
    }
  }

  // reduction sum
  constexpr uint32_t FULL_MASK = 0xffffffff;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
    }
  }

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
    int pad_thread_offset = ((first_batch + i) / pad_batch_stride) * stride +
                            ELEMENTS_PER_LDG_STG * local_idx;
    const uint8_t *curr_mask = pad_mask + pad_thread_offset;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        output_t out[ELEMENTS_PER_LDG_STG];
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = (output_reg[i][it + element] *
                          (grad_reg[i][it + element] - sum[i]));
        }
        // store them in global memory
        int itr_jmp = it * WARP_SIZE;
        int itr_idx = i * element_count + itr_jmp;
        // It is kind of unfortunate this has to be here to zero something out
        // that is close to zero in the first place
        apply_mask<input_t, ELEMENTS_PER_LDG_STG>(&out[0], 0.0,
                                                  curr_mask + itr_jmp);
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(gradInput + itr_idx, out);
      }
    }
  }
}

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t>
using masked_softmax_backward_func =
    void (*)(output_t *gradInput, const input_t *grad, const input_t *output,
             const uint8_t *pad_mask, int batch_size, int stride,
             int element_count, int pad_batch_stride);

template <typename input_t, typename output_t, typename acc_t>
bool warp_masked_softmax_backward_kernel(
    int log2_elements, int &warp_size, int &batches_per_warp,
    masked_softmax_backward_func<input_t, output_t> &kernel) {
  // determine size of a warp
  const int next_power_of_two = 1 << log2_elements;
  warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

  // determine how many batches a warp should process.
  batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  switch (log2_elements) {
  case 0: // 1
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 1, 1>;
    break;
  case 1: // 2
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 2, 1>;
    break;
  case 2: // 4
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 4, 1>;
    break;
  case 3: // 8
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 8, 1>;
    break;
  case 4: // 16
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 16, 1>;
    break;
  case 5: // 32
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 1, 32, 1>;
    break;
  case 6: // 64
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 2, 32, 1>;
    break;
  case 7: // 128
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 2, 4, 32, 1>;
    break;
  case 8: // 256
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 1, 8, 32, 1>;
    break;
  case 9: // 512
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 1, 16, 32, 1>;
    break;
  case 10: // 1024
    kernel =
        &masked_softmax_warp_backward<input_t, output_t, acc_t, 1, 32, 32, 1>;
    break;
  default:
    return false;
  }
  return true;
}

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_masked_softmax_backward(output_t *grad_input, const input_t *grad,
                                      const input_t *output,
                                      const uint8_t *pad_mask,
                                      int softmax_elements,
                                      int softmax_elements_stride,
                                      int batch_count, int pad_batch_stride) {
  if (softmax_elements == 0) {
    return true;
  } else if (softmax_elements <= 1024) {
    // compute function index. there's a function for each power of two size up
    // to 1024.
    int log2_elements = 0;
    while ((1 << log2_elements) < softmax_elements)
      ++log2_elements;

    masked_softmax_backward_func<input_t, output_t> kernel;
    int warp_size, batches_per_warp;
    if (!warp_masked_softmax_backward_kernel<input_t, output_t, acc_t>(
            log2_elements, warp_size, batches_per_warp, kernel)) {
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
    kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_input, grad, output, pad_mask, batch_count,
        softmax_elements_stride, softmax_elements, pad_batch_stride);
    return true;
  }
  return false;
}
} // namespace
