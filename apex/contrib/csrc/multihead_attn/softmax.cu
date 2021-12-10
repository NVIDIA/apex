#include "softmax.cuh"
#include "philox.cuh"
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <curand_kernel.h>

#include <assert.h>
#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>
#include <limits>
#include <stdint.h>

namespace {

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
} // namespace
