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
__global__ void fused_rope_forward(int sq, int b, int np, int hn, int hn2,
                                   const scalar_t* src, const scalar_t* cos,
                                   const scalar_t* sin, scalar_t* dst) {
  int sq_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = sq_id * b * np * hn + b_id * np * hn;
#pragma unroll
  for (int hn_id = threadIdx.x; hn_id < hn2; hn_id += blockDim.x) {
    scalar_t v_cos = cos[sq_id * hn2 + hn_id];
    scalar_t v_sin = sin[sq_id * hn2 + hn_id];
#pragma unroll
    for (int head_id = 0; head_id < np; head_id += 1) {
      int offset_src_dst = offset_block + head_id * hn + hn_id;
      scalar_t v_src = src[offset_src_dst];
      scalar_t v_src_rotate = (hn_id + hn2 / 2 < hn2)
                                  ? -src[offset_src_dst + hn2 / 2]
                                  : src[offset_src_dst + hn2 / 2 - hn2];
      dst[offset_src_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (hn > hn2) {
#pragma unroll
    for (int head_id = 0; head_id < np; head_id += 1) {
      int offset_head = offset_block + head_id * hn;
#pragma unroll
      for (int hn_id = hn2 + threadIdx.x; hn_id < hn; hn_id += blockDim.x) {
        int offset_src_dst = offset_head + hn_id;
        dst[offset_src_dst] = src[offset_src_dst];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_backward(int sq, int b, int np, int hn, int hn2,
                                    const scalar_t* src, const scalar_t* cos,
                                    const scalar_t* sin, scalar_t* dst) {
  int sq_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = sq_id * b * np * hn + b_id * np * hn;
#pragma unroll
  for (int hn_id = threadIdx.x; hn_id < hn2; hn_id += blockDim.x) {
    scalar_t v_cos = cos[sq_id * hn2 + hn_id];
    scalar_t v_sin = (hn_id + hn2 / 2 < hn2)
                         ? sin[sq_id * hn2 + hn_id + hn2 / 2]
                         : -sin[sq_id * hn2 + hn_id + hn2 / 2 - hn2];
#pragma unroll
    for (int head_id = 0; head_id < np; head_id += 1) {
      int offset_src_dst = offset_block + head_id * hn + hn_id;
      scalar_t v_src = src[offset_src_dst];
      scalar_t v_src_rotate = (hn_id + hn2 / 2 < hn2)
                                  ? src[offset_src_dst + hn2 / 2]
                                  : src[offset_src_dst + hn2 / 2 - hn2];
      dst[offset_src_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (hn > hn2) {
#pragma unroll
    for (int head_id = 0; head_id < np; head_id += 1) {
      int offset_head = offset_block + head_id * hn;
#pragma unroll
      for (int hn_id = hn2 + threadIdx.x; hn_id < hn; hn_id += blockDim.x) {
        dst[offset_head + hn_id] = 1.0;
      }
    }
  }
}

}  // end of anonymous namespace

template <typename scalar_t>
void dispatch_fused_rope_forward(int sq, int b, int np, int hn, int hn2,
                                 const scalar_t* input, const scalar_t* cos,
                                 const scalar_t* sin, scalar_t* output) {
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int threads_per_block = 256;
  dim3 blocks(sq, b);
  dim3 threads(threads_per_block);

  fused_rope_forward<<<blocks, threads, 0, stream>>>(sq, b, np, hn, hn2, input,
                                                     cos, sin, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_fused_rope_backward(int sq, int b, int np, int hn, int hn2,
                                  const scalar_t* output_grads,
                                  const scalar_t* cos, const scalar_t* sin,
                                  scalar_t* input_grads) {
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int threads_per_block = 256;
  dim3 blocks(sq, b);
  dim3 threads(threads_per_block);

  fused_rope_backward<<<blocks, threads, 0, stream>>>(
      sq, b, np, hn, hn2, output_grads, cos, sin, input_grads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
