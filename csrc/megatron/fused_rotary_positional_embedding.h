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
__global__ void fused_rope_forward(int h, int d, int d2, int stride_s,
                                   int stride_b, int stride_h, int stride_d,
                                   int o_stride_s, int o_stride_b,
                                   int o_stride_h, int o_stride_d,
                                   const scalar_t* src, const scalar_t* cos,
                                   const scalar_t* sin, scalar_t* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t v_cos = cos[s_id * d2 + d_id];
    scalar_t v_sin = sin[s_id * d2 + d_id];
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t v_src = src[offset_src];
      scalar_t v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? -src[offset_src + (d2 / 2) * stride_d]
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

template <typename scalar_t>
__global__ void fused_rope_backward(int h, int d, int d2, int stride_s,
                                    int stride_b, int stride_h, int stride_d,
                                    int o_stride_s, int o_stride_b,
                                    int o_stride_h, int o_stride_d,
                                    const scalar_t* src, const scalar_t* cos,
                                    const scalar_t* sin, scalar_t* dst) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t v_cos = cos[s_id * d2 + d_id];
    scalar_t v_sin = (d_id + d2 / 2 < d2)
                         ? sin[s_id * d2 + d_id + d2 / 2]
                         : -sin[s_id * d2 + d_id + d2 / 2 - d2];
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
        dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
      }
    }
  }
}

}  // end of anonymous namespace

template <typename scalar_t>
void dispatch_fused_rope_forward(int s, int b, int h, int d, int d2,
                                 int stride_s, int stride_b, int stride_h,
                                 int stride_d, int o_stride_s, int o_stride_b,
                                 int o_stride_h, int o_stride_d,
                                 const scalar_t* input, const scalar_t* cos,
                                 const scalar_t* sin, scalar_t* output) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_forward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
      o_stride_h, o_stride_d, input, cos, sin, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_fused_rope_backward(int s, int b, int h, int d, int d2,
                                  int stride_s, int stride_b, int stride_h,
                                  int stride_d, int o_stride_s, int o_stride_b,
                                  int o_stride_h, int o_stride_d,
                                  const scalar_t* output_grads,
                                  const scalar_t* cos, const scalar_t* sin,
                                  scalar_t* input_grads) {
  auto stream = at::cuda::getCurrentCUDAStream();

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(C10_WARP_SIZE, warps_per_block);

  fused_rope_backward<<<blocks, threads, 0, stream>>>(
      h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
      o_stride_h, o_stride_d, output_grads, cos, sin, input_grads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
