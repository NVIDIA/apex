/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do { \
  cudaError_t status_ = call; \
  if( status_ != cudaSuccess ) { \
    fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
    exit(1); \
  } \
} while(0)

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ __host__ int div_up(int m, int n) {
  return (m + n-1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ __host__ float sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void spin_wait_(int *barrier, int step, int expected) {

  // THE FOLLOWING CODE MUST BE EXECUTED BY A SINGLE THREAD IN THE CTA.

  // Update the global counter. Make sure prior writes are visible.
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(barrier), "r"(step));

  // Busy wait. We could use found = old + step with old = atomicAdd(...) but it's not faster.
  for( volatile int found = -1; found != expected; ) {
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(found) : "l"(barrier));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Input type followed by parameter type
enum PrecisionMode {
  FP32IOFP16W,
  FP32IOBF16W,
  FP32IOFP32W,
  FP16IOFP16W,
  FP16IOBF16W,
  FP16IOFP32W,
  BF16IOFP16W,
  BF16IOBF16W,
  BF16IOFP32W,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Group_sums {
  // Is it the 1st element of the group?
  int flag;
  // The sum.
  float sum;
  // The sum of squares.
  float sum_sq;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Group_sums_op {
  inline __device__ Group_sums operator()(const Group_sums &a, const Group_sums &b) {
    Group_sums dst;
    dst.sum    = b.flag ? b.sum    : (a.sum    + b.sum);
    dst.sum_sq = b.flag ? b.sum_sq : (a.sum_sq + b.sum_sq);
    dst.flag   = a.flag + b.flag;
    return dst;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Group_norm_nhwc_fwd_params {

  // The output buffer. Layout NHWC.
  void *y;
  // The sums for the bwd pass. Not written if it is a nullptr.
  float2 *sums;
  // The input buffer. Layout NHWC.
  const void *x;
  // The gamma scaling factor.
  const void *gamma;
  // The beta term to add in GN.
  const void *beta;
  // The constant epsilon for sqrt(var + epsilon).
  float epsilon;
  // The barriers for the persistent kernel.
  int *barriers;
  // The extra storage for multi-CTA reductions as well as to pass data to the bwd.
  float *red_buffer, *zeroed_red_buffer;

  // The number of instances in the batch.
  int n;
  // The height and width of each activation map. The number of channels.
  int h, w, c, hw, hwc;
  // The number of groups.
  int groups;
  // Do we apply the Swish activation function?
  bool with_swish;

  // Precomputed values and parameters to control the execution of the kernels.

  // The number of batch instances per block.
  int instances_per_block;
  // The number of activations computed per block.
  int acts_per_block;
  // The number of groups in each block.
  int groups_per_block;
  // The number of channels per group = c / groups.
  int channels_per_group; 
  // The number of channels per block = groups_per_block * channels_per_group.
  int channels_per_block;
  // The inverse of hwc in floats (to compute mean/var).
  float inv_hwc_per_group;
  // IO precision
  PrecisionMode precision;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_fwd_two_passes_setup(Group_norm_nhwc_fwd_params&, 
                                          size_t &red_buffer_elts);

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_fwd_two_passes_sum  (const Group_norm_nhwc_fwd_params&, cudaStream_t);

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_fwd_two_passes_scale(const Group_norm_nhwc_fwd_params&, cudaStream_t);

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Group_norm_nhwc_bwd_params {

  // The output buffer. Layout NHWC.
  void *dx;
  // The output buffer. Layout NHWC.
  void *dgamma;
  // The output buffer. Layout NHWC.
  void *dbeta;
  // The input buffer. Layout NHWC.
  const void *dy;
  // The input buffer. Layout NHWC.
  const void *x;
  // The gamma scaling factor.
  const void *gamma;
  // The beta term to add in GN.
  const void *beta;
  // The sums from the fwd pass.
  const float2 *sums;
  // The constant epsilon for sqrt(var + epsilon).
  float epsilon;
  // The barriers for the persistent kernel.
  int *barriers;
  // The extra storage for multi-CTA reductions as well as to pass data to the bwd.
  float *red_buffer, *zeroed_red_buffer;

  // The number of instances in the batch.
  int n;
  // The height and width of each activation map. The number of channels.
  int h, w, c, hw, hwc;
  // The number of groups.
  int groups;
  // Do we apply the Swish activation function?
  bool with_swish;

  // Precomputed values and parameters to control the execution of the kernels.

  // The number of batch instances per block.
  int instances_per_block;
  // The number of activations computed per block.
  int acts_per_block;
  // The number of groups in each block.
  int groups_per_block;
  // The number of channels per group = c / groups.
  int channels_per_group; 
  // The number of channels per block = groups_per_block * channels_per_group.
  int channels_per_block;
  // The inverse of hwc in floats (to compute mean/var).
  float inv_hwc_per_group;
  // IO precision
  PrecisionMode precision;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_bwd_two_passes_setup(Group_norm_nhwc_bwd_params&, 
                                          size_t &red_buffer_elts);

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_bwd_two_passes_sum  (const Group_norm_nhwc_bwd_params&, cudaStream_t);

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_bwd_two_passes_scale(const Group_norm_nhwc_bwd_params&, cudaStream_t);

////////////////////////////////////////////////////////////////////////////////////////////////////

