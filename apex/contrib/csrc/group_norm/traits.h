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

struct Fp32
{
  // I/O type is float32_t
  using IOType = float;
  // Doubled I/O type
  using IOType2 = float2;

  // Unpack input to accumulators type
  static inline __device__ float2 unpack(const float2& f2)
  {
    return f2;
  }

  // Pack the accumulators into outputs.
  static inline __device__ float2 pack(const float2& f2)
  {
    return f2;
  }

  static inline __device__ float2 zero()
  {
    return {0.f, 0.f};
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fp16
{
  // I/O type is __half
  using IOType = __half;
  // Doubled I/O type
  using IOType2 = __half2;

  // Unpack input to accumulators type
  static inline __device__ float2 unpack(const __half2& h2)
  {
    // FIXME(nkorobov): __half22float2 makes compilation error in container
    return {__half2float(h2.x), 
            __half2float(h2.y)};
  }

  // Pack the accumulators into outputs.
  static inline __device__ __half2 pack(const float2& f2)
  {
    // FIXME(nkorobov): __float22half2_rn makes compilation error in container
    return {__float2half_rn(f2.x), __float2half_rn(f2.y)};
  }

  static inline __device__ __half2 zero()
  {
    uint32_t zero = 0;
    return *reinterpret_cast<__half2*>(&zero);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Bf16
{
  // I/O type is __nv_bfloat16
  using IOType = __nv_bfloat16;
  // Doubled I/O type
  using IOType2 = __nv_bfloat162;

  // Unpack input to accumulators type
  static inline __device__ float2 unpack(const __nv_bfloat162& h2)
  {
    // FIXME(nkorobov): __half22float2 makes compilation error in container
    return {__bfloat162float(h2.x), 
            __bfloat162float(h2.y)};
  }

  // Pack the accumulators into outputs.
  static inline __device__ __nv_bfloat162 pack(const float2& f2)
  {
    // FIXME(nkorobov): __float22bfloat162_rn makes compilation error in container
    return {__float2bfloat16_rn(f2.x),	__float2bfloat16_rn(f2.y)};
  }

  static inline __device__ __nv_bfloat162 zero()
  {
    uint32_t zero = 0;
    return *reinterpret_cast<__nv_bfloat162*>(&zero);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


