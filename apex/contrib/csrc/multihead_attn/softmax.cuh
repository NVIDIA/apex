#pragma once
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
template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<__half, 1>(__half *dst,
                                                  const __half *src);

template <>
__device__ __inline__ void copy_vector<float, 1>(float *dst, const float *src);

template <>
__device__ __inline__ void copy_vector<__half, 4>(__half *dst,
                                                  const __half *src);
template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst,
                                                   const uint8_t *src);

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst,
                                                   const uint8_t *src);

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void apply_mask(Datatype *dst, Datatype value,
                                      const uint8_t *src);

template <>
__device__ __inline__ void apply_mask<__half, 1>(__half *dst, __half value,
                                                 const uint8_t *src);
template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void apply_additive_mask(Datatype *dst,
                                               const Datatype *additive_mask);
template <>
__device__ __inline__ void
apply_additive_mask<__half, 1>(__half *dst, const __half *additive_mask);
template <>
__device__ __inline__ void
apply_additive_mask<__half, 4>(__half *dst, const __half *additive_mask);
} // namespace

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
                                     int element_count);

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
                         softmax_forward_func<input_t, output_t> &kernel);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax(output_t *dst, const input_t *src, int softmax_elements,
                      int softmax_elements_stride, int batch_count);

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE, int ELEMENTS_PER_LDG_STG>
__global__ void additive_masked_softmax_dropout_warp_forward_vec4(
    output_t *dst, uint8_t *dropout_mask, const input_t *src,
    const input_t *pad_mask, int batch_size, int stride, int element_count,
    int pad_batch_stride, at::PhiloxCudaState philox_args, float p);

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE, int ELEMENTS_PER_LDG_STG>
__global__ void additive_masked_softmax_dropout_warp_forward(
    output_t *dst, uint8_t *dropout_mask, const input_t *src,
    const input_t *pad_mask, int batch_size, int stride, int element_count,
    int pad_batch_stride, at::PhiloxCudaState philox_args, float p);

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
        &kernel);

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
    masked_softmax_backward_func<input_t, output_t> &kernel);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_additive_masked_softmax_dropout(
    output_t *dst, uint8_t *dropout_mask, const input_t *src,
    const input_t *pad_mask, int totalElements, int softmax_elements,
    int softmax_elements_stride, int batch_count, int pad_batch_stride, float p,
    cudaStream_t streamid); // p is the probability to keep, not drop

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void additive_masked_softmax_warp_forward(
    input_t *dst, const output_t *src, const input_t *pad_mask, int batch_size,
    int stride, int element_count, int pad_batch_stride);

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
    additive_masked_softmax_forward_func<input_t, output_t> &kernel);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_additive_masked_softmax(output_t *dst, const input_t *src,
                                      const input_t *pad_mask,
                                      int softmax_elements,
                                      int softmax_elements_stride,
                                      int batch_count, int pad_batch_stride);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_additive_masked_softmax_stream(
    output_t *dst, const input_t *src, const input_t *pad_mask,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    int pad_batch_stride, cudaStream_t streamid);

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void
masked_softmax_warp_forward(input_t *dst, const output_t *src,
                            const uint8_t *pad_mask, int batch_size, int stride,
                            int element_count, int pad_batch_stride);

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
    masked_softmax_forward_func<input_t, output_t> &kernel);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_masked_softmax(output_t *dst, const input_t *src,
                             const uint8_t *pad_mask, int softmax_elements,
                             int softmax_elements_stride, int batch_count,
                             int pad_batch_stride);

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate
// over all data. WARP_SIZE number of elements working on a single batch, has to
// be a power of two. ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void time_masked_softmax_warp_forward(
    input_t *dst, const output_t *src, const uint8_t *pad_mask, int batch_size,
    int stride, int element_count, int mod_seq_len);

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
    time_masked_softmax_forward_func<input_t, output_t> &kernel);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_time_masked_softmax(output_t *dst, const input_t *src,
                                  const uint8_t *pad_mask, int softmax_elements,
                                  int softmax_elements_stride, int batch_count,
                                  int mod_seq_len);

int log2_ceil_native(int value);

template <typename T>
__device__ __forceinline__ T
WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize,
                     unsigned int mask = 0xffffffff);

template <typename acc_t, int WARP_BATCH, int WARP_SIZE>
__device__ __forceinline__ void warp_reduce_sum(acc_t *sum);

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements, bool is_log_softmax>
__global__ void masked_scale_softmax_warp_backward_masked_dgrad(
    output_t *gradInput, const input_t *grad, const input_t *output,
    const uint8_t *mask, const uint8_t *pad_mask, acc_t scale, int batch_size,
    int stride, int element_count, int heads);

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_masked_scale_softmax_backward_masked_out(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const uint8_t *mask, const uint8_t *pad_mask, acc_t scale,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    int heads);

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_masked_scale_softmax_backward_masked_out_stream(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const uint8_t *mask, const uint8_t *pad_mask, acc_t scale,
    int softmax_elements, int softmax_elements_stride, int batch_count,
    int heads, cudaStream_t streamid);

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements, bool is_log_softmax>
__global__ void
masked_scale_softmax_warp_backward(output_t *gradInput, const input_t *grad,
                                   const input_t *output, const uint8_t *mask,
                                   acc_t scale, int batch_size, int stride,
                                   int element_count);

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG,
          bool is_log_softmax>
__global__ void masked_scale_softmax_warp_backward_recompute(
    output_t *gradInput, const input_t *grad, const input_t *softmax_input,
    const input_t *pad_mask, const uint8_t *mask, acc_t scale, int batch_size,
    int stride, int pad_batch_stride, int element_count);

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
                                                      is_log_softmax> &kernel);

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
bool dispatch_masked_scale_softmax_backward_recompute(
    output_t *grad_input, const input_t *grad, const input_t *softmax_input,
    const input_t *pad_mask, const uint8_t *mask, acc_t scale,
    int softmax_elements, int softmax_elements_stride, int pad_batch_stride,
    int batch_count, cudaStream_t streamid);

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_masked_scale_softmax_backward_stream(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const uint8_t *mask, acc_t scale, int softmax_elements,
    int softmax_elements_stride, int batch_count, cudaStream_t streamid);

// elementwise multiplication called in at::softmax_backward_data is fused
// inside softmax dgrad kernel as a result of fusion, intermediate
// multiplication result is stored in fp32 in registers, instead of fp16
template <typename input_t, typename output_t, typename acc_t,
          int log2_elements, bool is_log_softmax>
__global__ void
softmax_warp_backward_fused_native(output_t *gradInput, const input_t *grad,
                                   const input_t *output, int batch_size,
                                   int stride, int element_count);

template <typename input_t, typename output_t, typename acc_t,
          bool is_log_softmax>
void dispatch_softmax_backward_fused_native(
    output_t *grad_input, const input_t *grad, const input_t *output,
    int softmax_elements, int softmax_elements_stride, int batch_count);

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void softmax_warp_backward(__half *gradInput, const __half *grad,
                                      const __half *output, int batch_size,
                                      int stride, int element_count);

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
    softmax_backward_func<input_t, output_t> &kernel);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax_backward(output_t *grad_input, const input_t *grad,
                               const input_t *output, int softmax_elements,
                               int softmax_elements_stride, int batch_count);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_softmax_backward_stream(output_t *grad_input, const input_t *grad,
                                      const input_t *output,
                                      int softmax_elements,
                                      int softmax_elements_stride,
                                      int batch_count, cudaStream_t streamid);

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH,
          int WARP_ITERATIONS, int WARP_SIZE = 32, int ELEMENTS_PER_LDG_STG = 1>
__global__ void
masked_softmax_warp_backward(__half *gradInput, const __half *grad,
                             const __half *output, const uint8_t *pad_mask,
                             int batch_size, int stride, int element_count,
                             int pad_batch_stride);

template <typename input_t, typename output_t, typename acc_t>
bool dispatch_masked_softmax_backward(output_t *grad_input, const input_t *grad,
                                      const input_t *output,
                                      const uint8_t *pad_mask,
                                      int softmax_elements,
                                      int softmax_elements_stride,
                                      int batch_count, int pad_batch_stride);
