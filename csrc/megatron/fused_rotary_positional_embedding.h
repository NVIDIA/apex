/* coding=utf-8
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__device__ void fused_rope_block_forward(
    const scalar_t* src, const float* freqs, scalar_t* dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos, v_sin;
    sincosf(freqs[s_id * d2 + d_id], &v_sin, &v_cos);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t v_src = src[offset_src];
      scalar_t v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? -src[offset_src + (d2 / 2) * stride_d]
                                  : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] =
          v_src * (scalar_t)v_cos + v_src_rotate * (scalar_t)v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_rope_block_backward(
    const scalar_t* src, const float* freqs, scalar_t* dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t v_cos = cosf(freqs[s_id * d2 + d_id]);
    scalar_t v_sin = (d_id + d2 / 2 < d2)
                         ? sinf(freqs[s_id * d2 + d_id + d2 / 2])
                         : -sinf(freqs[s_id * d2 + d_id + d2 / 2 - d2]);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t v_src = src[offset_src];
      scalar_t v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? src[offset_src + (d2 / 2) * stride_d]
                                  : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_forward(const int h, const int d, const int d2,
                                   const int stride_s, const int stride_b,
                                   const int stride_h, const int stride_d,
                                   const int o_stride_s, const int o_stride_b,
                                   const int o_stride_h, const int o_stride_d,
                                   const scalar_t* src, const float* freqs,
                                   scalar_t* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_forward(src, freqs, dst, offset_block, offset_block_dst, h,
                           d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward(const int h, const int d, const int d2,
                                    const int stride_s, const int stride_b,
                                    const int stride_h, const int stride_d,
                                    const int o_stride_s, const int o_stride_b,
                                    const int o_stride_h, const int o_stride_d,
                                    const scalar_t* src, const float* freqs,
                                    scalar_t* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_backward(src, freqs, dst, offset_block, offset_block_dst, h,
                            d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t_0, typename scalar_t_1>
__device__ void fused_rope_cached_block_forward(
    const scalar_t_0* src, const scalar_t_1* cos, const scalar_t_1* sin,
    scalar_t_0* dst, const int s_id, const int offset_block,
    const int offset_block_dst, const int h, const int d, const int d2,
    const int stride_h, const int stride_d, const int o_stride_h,
    const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t_0 v_cos = cos[s_id * d2 + d_id];
    scalar_t_0 v_sin = sin[s_id * d2 + d_id];
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t_0 v_src = src[offset_src];
      scalar_t_0 v_src_rotate =
          (d_id + d2 / 2 < d2) ? -src[offset_src + (d2 / 2) * stride_d]
                               : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t_0, typename scalar_t_1>
__device__ void fused_rope_cached_block_backward(
    const scalar_t_0* src, const scalar_t_1* cos, const scalar_t_1* sin,
    scalar_t_0* dst, const int s_id, const int offset_block,
    const int offset_block_dst, const int h, const int d, const int d2,
    const int stride_h, const int stride_d, const int o_stride_h,
    const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t_0 v_cos = cos[s_id * d2 + d_id];
    scalar_t_0 v_sin = (d_id + d2 / 2 < d2)
                           ? sin[s_id * d2 + d_id + d2 / 2]
                           : -sin[s_id * d2 + d_id + d2 / 2 - d2];
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t_0 v_src = src[offset_src];
      scalar_t_0 v_src_rotate =
          (d_id + d2 / 2 < d2) ? src[offset_src + (d2 / 2) * stride_d]
                               : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t_0, typename scalar_t_1>
__global__ void fused_rope_cached_forward(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, const scalar_t_0* src, const scalar_t_1* cos,
    const scalar_t_1* sin, scalar_t_0* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_cached_block_forward(src, cos, sin, dst, s_id, offset_block,
                                  offset_block_dst, h, d, d2, stride_h,
                                  stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t_0, typename scalar_t_1>
__global__ void fused_rope_cached_backward(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, const scalar_t_0* src, const scalar_t_1* cos,
    const scalar_t_1* sin, scalar_t_0* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_cached_block_backward(src, cos, sin, dst, s_id, offset_block,
                                   offset_block_dst, h, d, d2, stride_h,
                                   stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_thd_forward(
    const int h, const int d, const int d2, const int stride_t,
    const int stride_h, const int stride_d, const int o_stride_t,
    const int o_stride_h, const int o_stride_d, const scalar_t* src,
    const int* cu_seqlens, const float* freqs, scalar_t* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int t_id = s_id + cu_seqlens[b_id];
  if (t_id >= cu_seqlens[b_id + 1]) return;
  int offset_block = t_id * stride_t;
  int offset_block_dst = t_id * o_stride_t;
  fused_rope_block_forward(src, freqs, dst, offset_block, offset_block_dst, h,
                           d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_thd_backward(
    const int h, const int d, const int d2, const int stride_t,
    const int stride_h, const int stride_d, const int o_stride_t,
    const int o_stride_h, const int o_stride_d, const scalar_t* src,
    const int* cu_seqlens, const float* freqs, scalar_t* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int t_id = s_id + cu_seqlens[b_id];
  if (t_id >= cu_seqlens[b_id + 1]) return;
  int offset_block = t_id * stride_t;
  int offset_block_dst = t_id * o_stride_t;
  fused_rope_block_backward(src, freqs, dst, offset_block, offset_block_dst, h,
                            d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t_0, typename scalar_t_1>
__global__ void fused_rope_2d_forward(
    const int ih, const int iw, const int h, const int d, const int stride_b,
    const int stride_ih, const int stride_iw, const int stride_h,
    const int stride_d, const int o_stride_b, const int o_stride_s,
    const int o_stride_h, const int o_stride_d, const scalar_t_0* src,
    const scalar_t_1* cos_h, const scalar_t_1* sin_h, const scalar_t_1* cos_w,
    const scalar_t_1* sin_w, scalar_t_0* dst) {
  int ih_id = blockIdx.x, iw_id = blockIdx.y, b_id = blockIdx.z;
  // apply to height
  int offset_block = b_id * stride_b + ih_id * stride_ih + iw_id * stride_iw;
  int offset_block_dst = b_id * o_stride_b + (ih_id * iw + iw_id) * o_stride_s;
  int s_id = ih_id;  // for cos_h and sin_h
  fused_rope_cached_block_forward(src, cos_h, sin_h, dst, s_id, offset_block,
                                  offset_block_dst, h, d / 2, d / 2, stride_h,
                                  stride_d, o_stride_h, o_stride_d);
  // apply to width
  offset_block += d / 2 * stride_d;
  offset_block_dst += d / 2 * o_stride_d;
  s_id = iw_id;  // for cos_w and sin_w
  fused_rope_cached_block_forward(src, cos_w, sin_w, dst, s_id, offset_block,
                                  offset_block_dst, h, d / 2, d / 2, stride_h,
                                  stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t_0, typename scalar_t_1>
__global__ void fused_rope_2d_backward(
    const int ih, const int iw, const int h, const int d, const int stride_b,
    const int stride_ih, const int stride_iw, const int stride_h,
    const int stride_d, const int o_stride_b, const int o_stride_s,
    const int o_stride_h, const int o_stride_d, const scalar_t_0* src,
    const scalar_t_1* cos_h, const scalar_t_1* sin_h, const scalar_t_1* cos_w,
    const scalar_t_1* sin_w, scalar_t_0* dst) {
  int ih_id = blockIdx.x, iw_id = blockIdx.y, b_id = blockIdx.z;
  // apply to height
  int offset_block = b_id * stride_b + ih_id * stride_ih + iw_id * stride_iw;
  int offset_block_dst = b_id * o_stride_b + (ih_id * iw + iw_id) * o_stride_s;
  int s_id = ih_id;  // for cos_h and sin_h
  fused_rope_cached_block_backward(src, cos_h, sin_h, dst, s_id, offset_block,
                                   offset_block_dst, h, d / 2, d / 2, stride_h,
                                   stride_d, o_stride_h, o_stride_d);
  // apply to width
  offset_block += d / 2 * stride_d;
  offset_block_dst += d / 2 * o_stride_d;
  s_id = iw_id;  // for cos_w and sin_w
  fused_rope_cached_block_backward(src, cos_w, sin_w, dst, s_id, offset_block,
                                   offset_block_dst, h, d / 2, d / 2, stride_h,
                                   stride_d, o_stride_h, o_stride_d);
}

}  // end of anonymous namespace

template <typename scalar_t>
void dispatch_fused_rope_forward(const int s, const int b, const int h,
                                 const int d, const int d2, const int stride_s,
                                 const int stride_b, const int stride_h,
                                 const int stride_d, const int o_stride_s,
                                 const int o_stride_b, const int o_stride_h,
                                 const int o_stride_d, const scalar_t* input,
                                 const float* freqs, scalar_t* output) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_forward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
      o_stride_h, o_stride_d, input, freqs, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_fused_rope_backward(const int s, const int b, const int h,
                                  const int d, const int d2, const int stride_s,
                                  const int stride_b, const int stride_h,
                                  const int stride_d, const int o_stride_s,
                                  const int o_stride_b, const int o_stride_h,
                                  const int o_stride_d,
                                  const scalar_t* output_grads,
                                  const float* freqs, scalar_t* input_grads) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_backward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
      o_stride_h, o_stride_d, output_grads, freqs, input_grads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t_0, typename scalar_t_1>
void dispatch_fused_rope_cached_forward(
    const int s, const int b, const int h, const int d, const int d2,
    const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, const scalar_t_0* input,
    const scalar_t_1* cos, const scalar_t_1* sin, scalar_t_0* output) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_cached_forward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
      o_stride_h, o_stride_d, input, cos, sin, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t_0, typename scalar_t_1>
void dispatch_fused_rope_cached_backward(
    const int s, const int b, const int h, const int d, const int d2,
    const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, const scalar_t_0* output_grads,
    const scalar_t_1* cos, const scalar_t_1* sin, scalar_t_0* input_grads) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_cached_backward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
      o_stride_h, o_stride_d, output_grads, cos, sin, input_grads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_fused_rope_thd_forward(const int max_s, const int b, const int h,
                                     const int d, const int d2,
                                     const int stride_t, const int stride_h,
                                     const int stride_d, const int o_stride_t,
                                     const int o_stride_h, const int o_stride_d,
                                     const scalar_t* input,
                                     const int* cu_seqlens, const float* freqs,
                                     scalar_t* output) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(max_s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_thd_forward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_t, stride_h, stride_d, o_stride_t, o_stride_h,
      o_stride_d, input, cu_seqlens, freqs, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_fused_rope_thd_backward(
    const int max_s, const int b, const int h, const int d, const int d2,
    const int stride_t, const int stride_h, const int stride_d,
    const int o_stride_t, const int o_stride_h, const int o_stride_d,
    const scalar_t* output_grads, const int* cu_seqlens, const float* freqs,
    scalar_t* input_grads) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(max_s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_thd_backward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_t, stride_h, stride_d, o_stride_t, o_stride_h,
      o_stride_d, output_grads, cu_seqlens, freqs, input_grads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t_0, typename scalar_t_1>
void dispatch_fused_rope_2d_forward(
    const int b, const int ih, const int iw, const int h, const int d,
    const int stride_b, const int stride_ih, const int stride_iw,
    const int stride_h, const int stride_d, const int o_stride_b,
    const int o_stride_s, const int o_stride_h, const int o_stride_d,
    const scalar_t_0* input, const scalar_t_1* cos_h, const scalar_t_1* sin_h,
    const scalar_t_1* cos_w, const scalar_t_1* sin_w, scalar_t_0* output) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(ih, iw, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_2d_forward<<<blocks, threads, 0, stream>>>(
      ih, iw, h, d, stride_b, stride_ih, stride_iw, stride_h, stride_d,
      o_stride_b, o_stride_s, o_stride_h, o_stride_d, input, cos_h, sin_h,
      cos_w, sin_w, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t_0, typename scalar_t_1>
void dispatch_fused_rope_2d_backward(
    const int b, const int ih, const int iw, const int h, const int d,
    const int stride_b, const int stride_ih, const int stride_iw,
    const int stride_h, const int stride_d, const int o_stride_b,
    const int o_stride_s, const int o_stride_h, const int o_stride_d,
    const scalar_t_0* output_grads, const scalar_t_1* cos_h,
    const scalar_t_1* sin_h, const scalar_t_1* cos_w, const scalar_t_1* sin_w,
    scalar_t_0* input_grads) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(ih, iw, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_2d_backward<<<blocks, threads, 0, stream>>>(
      ih, iw, h, d, stride_b, stride_ih, stride_iw, stride_h, stride_d,
      o_stride_b, o_stride_s, o_stride_h, o_stride_d, output_grads, cos_h,
      sin_h, cos_w, sin_w, input_grads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
