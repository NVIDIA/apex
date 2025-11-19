/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fp32 {
  // Type is float32_t
  using Type = float;
  // Doubled type
  using Type2 = float2;

  // Unpack input to accumulators type
  static inline __device__ float2 unpack(const float2& f2) { return f2; }

  // Pack the accumulators into outputs.
  static inline __device__ float2 pack(const float2& f2) { return f2; }

  static inline __device__ float2 zero() { return {0.f, 0.f}; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fp16 {
  // Type is __half
  using Type = __half;
  // Doubled type
  using Type2 = __half2;

  // Unpack input to accumulators type
  static inline __device__ float2 unpack(const __half2& h2) {
    // FIXME(nkorobov): __half22float2 makes compilation error in container
    return {__half2float(h2.x), __half2float(h2.y)};
  }

  // Pack the accumulators into outputs.
  static inline __device__ __half2 pack(const float2& f2) {
    // FIXME(nkorobov): __float22half2_rn makes compilation error in container
    return {__float2half_rn(f2.x), __float2half_rn(f2.y)};
  }

  static inline __device__ __half2 zero() {
    uint32_t zero = 0;
    return *reinterpret_cast<__half2*>(&zero);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Bf16 {
  // Type is __nv_bfloat16
  using Type = __nv_bfloat16;
  // Doubled type
  using Type2 = __nv_bfloat162;

  // Unpack input to accumulators type
  static inline __device__ float2 unpack(const __nv_bfloat162& h2) {
    // FIXME(nkorobov): __half22float2 makes compilation error in container
    return {__bfloat162float(h2.x), __bfloat162float(h2.y)};
  }

  // Pack the accumulators into outputs.
  static inline __device__ __nv_bfloat162 pack(const float2& f2) {
    // FIXME(nkorobov): __float22bfloat162_rn makes compilation error in container
    return {__float2bfloat16_rn(f2.x), __float2bfloat16_rn(f2.y)};
  }

  static inline __device__ __nv_bfloat162 zero() {
    uint32_t zero = 0;
    return *reinterpret_cast<__nv_bfloat162*>(&zero);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
struct Fp32IOFp16W {
  // IO traits
  using IOTraits = Fp32;
  // Weigths traits
  using WTraits = Fp16;
};

struct Fp32IOBf16W {
  // IO traits
  using IOTraits = Fp32;
  // Weigths traits
  using WTraits = Bf16;
};

struct Fp32IOFp32W {
  // IO traits
  using IOTraits = Fp32;
  // Weigths traits
  using WTraits = Fp32;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fp16IOFp16W {
  // IO traits
  using IOTraits = Fp16;
  // Weigths traits
  using WTraits = Fp16;
};

struct Fp16IOBf16W {
  // IO traits
  using IOTraits = Fp16;
  // Weigths traits
  using WTraits = Bf16;
};

struct Fp16IOFp32W {
  // IO traits
  using IOTraits = Fp16;
  // Weigths traits
  using WTraits = Fp32;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
struct Bf16IOFp16W {
  // IO traits
  using IOTraits = Bf16;
  // Weigths traits
  using WTraits = Fp16;
};

struct Bf16IOBf16W {
  // IO traits
  using IOTraits = Bf16;
  // Weigths traits
  using WTraits = Bf16;
};

struct Bf16IOFp32W {
  // IO traits
  using IOTraits = Bf16;
  // Weigths traits
  using WTraits = Fp32;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
