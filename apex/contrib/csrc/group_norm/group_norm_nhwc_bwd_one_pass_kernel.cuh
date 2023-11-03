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
#include "group_norm_nhwc.h"
#include "traits.h"
#include <assert.h>
#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// B A C K W A R D
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, int ACTS_PER_BLOCK_, int CHANNELS_PER_GROUP_, int THREADS_PER_BLOCK_ >
__global__ __launch_bounds__(THREADS_PER_BLOCK_) 
  void group_norm_nhwc_bwd_one_pass_kernel(Group_norm_nhwc_bwd_params params) {

  // The IO traits.
  using Traits = Traits_;
  // The IO traits.
  using IOTraits = typename Traits::IOTraits;
  // The Weights traits.
  using WTraits = typename Traits::WTraits;

  // The IO type
  using IOType = typename IOTraits::Type;
  // The IO doubled type
  using IOType2 = typename IOTraits::Type2;

  // Weights type
  using WType = typename WTraits::Type;
  // Weights doubled type
  using WType2 = typename WTraits::Type2;

  // The number of activations per block.
  constexpr int ACTS_PER_BLOCK = ACTS_PER_BLOCK_;
  // The number of channels per group.
  constexpr int CHANNELS_PER_GROUP = CHANNELS_PER_GROUP_;
  // The number of threads per block.
  constexpr int THREADS_PER_BLOCK = THREADS_PER_BLOCK_;
  // The number of channels per thread (load fp16x2 numbers).
  constexpr int CHANNELS_PER_THREAD = 2;

  // The number of threads needed per activation.
  constexpr int THREADS_PER_ACT = CHANNELS_PER_GROUP / CHANNELS_PER_THREAD;
  // The number of activations that are loaded per loop.
  constexpr int ACTS_PER_LOOP = THREADS_PER_BLOCK / THREADS_PER_ACT;
  // The number of rows per thread.
  constexpr int ACTS_PER_THREAD = (ACTS_PER_BLOCK + ACTS_PER_LOOP-1) / ACTS_PER_LOOP;

  // The number of active threads.
  constexpr int ACTIVE_THREADS = THREADS_PER_BLOCK / THREADS_PER_ACT * THREADS_PER_ACT;

  // The object in charge of doing the sums for the block.
  typedef cub::BlockReduce<float2, THREADS_PER_BLOCK> Block_reduce;
  // Allocate shared memory for Block_reduce.
  __shared__ typename Block_reduce::TempStorage temp_storage;
  // Allocate shared memory to store the sums.
  __shared__ float2 smem_sums;
  // Allocate shared memory to store the gamma/beta gradients.
  __shared__ float4 smem_dgamma_dbeta[THREADS_PER_BLOCK]; 

  // Shared memory to store the gradients for gamma and beta.

  // The first activation loaded by that thread.
  int hwi = blockIdx.x * params.acts_per_block + threadIdx.x / THREADS_PER_ACT;
  // The first channel loaded by that thread.
  int ci = threadIdx.x % THREADS_PER_ACT * CHANNELS_PER_THREAD;

  // Is it an active thread?
  const bool is_active = threadIdx.x < ACTIVE_THREADS;

  // Iterate over the iterms in the batch.
  for( int ngi = blockIdx.y, step = 0; ngi < params.n*params.groups; ngi += gridDim.y, ++step ) {

    // The instance and the group. TODO: Use fast divmod?
    int ni = ngi / params.groups;
    int gi = ngi % params.groups;

    // The sums from the fwd pass.
    float2 fwd = params.sums[ngi];
    // The mean of X (computed during the fwd pass -- one value per batch*group).
    float x_mean = fwd.x;
    // The mean of squares of X (computed during the fwd pass -- one value per batch*group).
    float x_sq_mean = fwd.y;
    // The variance.
    float x_var = x_sq_mean - x_mean * x_mean;
    // The reciprocal of the standard deviation (i.e. 1.f / sqrt(var + epsilon)).
    float rcp_x_stddev = x_var <= 0.f ? 1.f : 1.f / sqrtf(x_var + params.epsilon);

    // The offset to the first activation loaded by that thread.
    const int64_t offset = (int64_t) ni*params.hwc + gi*CHANNELS_PER_GROUP + ci;
    // The pointer to the first activation loaded by that thread.
    const IOType *x_ptr = &reinterpret_cast<const IOType*>(params.x)[offset];
    // The pointer to the first gradient loaded by that thread.
    const IOType *dy_ptr = &reinterpret_cast<const IOType*>(params.dy)[offset];

    // Load the X and dY into registers.
    IOType2 x[ACTS_PER_THREAD], dy[ACTS_PER_THREAD];
    #pragma unroll
    for( int ii = 0; ii < ACTS_PER_THREAD; ++ii ) {
      int hwj = hwi + ii*ACTS_PER_LOOP;
      x [ii] = IOTraits::zero();
      dy[ii] = IOTraits::zero();
      if( is_active && hwj < params.hw ) {
        x [ii] = *reinterpret_cast<const IOType2*>(&x_ptr [hwj*params.c]);
        dy[ii] = *reinterpret_cast<const IOType2*>(&dy_ptr[hwj*params.c]);
      }
    }

    // Load gamma as well.
    float2 gamma_f2 = make_float2(0.f, 0.f);
    float2 beta_f2 = make_float2(0.f, 0.f);
    if( is_active ) {
      gamma_f2 = WTraits::unpack(*reinterpret_cast<const WType2*>(
        &reinterpret_cast<const WType*>(params.gamma)[gi*CHANNELS_PER_GROUP+ci]));
      if (params.with_swish) {
        beta_f2 = WTraits::unpack(*reinterpret_cast<const WType2*>(
          &reinterpret_cast<const WType*>(params.beta) [gi*CHANNELS_PER_GROUP+ci]));
      }
    }

    // Gradients for gamma and beta (for this particular group).
    float4 dgamma_dbeta = make_float4(0.f, 0.f, 0.f, 0.f);
    // Accumulated gradients for dgrad calculation.
    float mean_1 = 0.f, mean_2 = 0.f;

    // Compute the sum and the sum of squares for each thread.
    #pragma unroll
    for( int ii = 0; ii < ACTS_PER_THREAD; ++ii ) {
      // Convert x to float.
      float2 x_f2  = IOTraits::unpack(x [ii]);
      // Convert dY to float.
      float2 dy_f2 = IOTraits::unpack(dy[ii]);

      // X - X_mean.
      float x_minus_x_mean_x = x_f2.x - x_mean;
      float x_minus_x_mean_y = x_f2.y - x_mean;

      // Normalize X.
      float x_norm_x = x_minus_x_mean_x * rcp_x_stddev;
      float x_norm_y = x_minus_x_mean_y * rcp_x_stddev;

      if (params.with_swish) {
        float x_gn_x = x_norm_x * gamma_f2.x + beta_f2.x;
        float x_gn_y = x_norm_y * gamma_f2.y + beta_f2.y;
        float s_x = sigmoid(x_gn_x);
        float s_y = sigmoid(x_gn_y);
        dy_f2.x = dy_f2.x * s_x * (1.f + x_gn_x * (1.f - s_x));
        dy_f2.y = dy_f2.y * s_y * (1.f + x_gn_y * (1.f - s_y));
      }

      // Update beta.
      dgamma_dbeta.z += dy_f2.x;
      dgamma_dbeta.w += dy_f2.y;

      // Update dgamma.
      dgamma_dbeta.x += dy_f2.x * x_norm_x;
      dgamma_dbeta.y += dy_f2.y * x_norm_y;

      // The gradient that enters the x_norm node.
      float dx_norm_x = dy_f2.x * gamma_f2.x;
      float dx_norm_y = dy_f2.y * gamma_f2.y;

      // Add to the 1st mean.
      mean_1 += dx_norm_x * x_norm_x;
      mean_1 += dx_norm_y * x_norm_y;

      // Add to the 2nd mean.
      mean_2 += dx_norm_x;
      mean_2 += dx_norm_y;
    }

    // Pack valid gradients.
    float2 sums = make_float2(0.f, 0.f);
    if( ACTIVE_THREADS == THREADS_PER_BLOCK || is_active ) {
      sums = make_float2(mean_1, mean_2);
    }

    // Store dgamma and dbeta to shared memory.
    smem_dgamma_dbeta[threadIdx.x] = dgamma_dbeta;

    // Compute the sums for the block.
    sums = Block_reduce(temp_storage).Reduce(sums, [](const float2 &a, const float2 &b) {
      return make_float2(a.x + b.x, a.y + b.y);
    });

    // Make sure we can read gamma/beta from smemory. Block_reduce uses one syncthread already.
    __syncthreads();

    // Compute gamma/beta for the block.
    if( threadIdx.x < THREADS_PER_ACT ) {
      for( int ii = 1; ii < ACTS_PER_LOOP; ++ii ) {
        float4 other = smem_dgamma_dbeta[threadIdx.x + ii*THREADS_PER_ACT];
        dgamma_dbeta.x += other.x;
        dgamma_dbeta.y += other.y;
        dgamma_dbeta.z += other.z;
        dgamma_dbeta.w += other.w;
      }
    }

    // The position in the channel dimension - 2 channels per thread.
    int cj = gi * THREADS_PER_ACT + threadIdx.x;
    // The reduction buffer dfor gamma/dbeta.
    float *red_buffer_dgamma_dbeta = &params.zeroed_red_buffer[cj];

    // The first threads store their gradients for gamma/beta.
    if( threadIdx.x < THREADS_PER_ACT ) {
      atomicAdd(&red_buffer_dgamma_dbeta[0*params.c/2], dgamma_dbeta.x);
      atomicAdd(&red_buffer_dgamma_dbeta[1*params.c/2], dgamma_dbeta.y);
      atomicAdd(&red_buffer_dgamma_dbeta[2*params.c/2], dgamma_dbeta.z);
      atomicAdd(&red_buffer_dgamma_dbeta[3*params.c/2], dgamma_dbeta.w);
    }

    // The block leader stores to global memory, if needed.
    if( gridDim.x > 1 ) {

      // The index of the buffer.
      int red_buffer_idx = step & 1;
      // The barrier.
      int *barrier = &params.barriers[red_buffer_idx*gridDim.y + blockIdx.y];
      // The offset to the reduction buffer.
      int red_buffer_offset = red_buffer_idx*gridDim.x*gridDim.y*2;
      // The reduction buffer.
      float2 *red_buffer = reinterpret_cast<float2*>(&params.red_buffer[red_buffer_offset]);

      // The offset to the reduction buffer for dgamma/dbeta.

      // The first thread stores its sums.
      if( threadIdx.x == 0 ) {
        red_buffer[blockIdx.x*gridDim.y + blockIdx.y] = sums;
      }

      // Make sure the data is in memory.
      if( threadIdx.x == 0 ) {
        spin_wait_(barrier, (step & 2) ? -1 : 1, (step & 2) ? 0 : gridDim.x);
      }
      __syncthreads();

      // Update the sums.
      for( int ii = 0; ii < gridDim.x; ++ii ) {
        if( ii != blockIdx.x && threadIdx.x == 0 ) {
          float2 other_sums = red_buffer[ii*gridDim.y + blockIdx.y];
          sums.x += other_sums.x;
          sums.y += other_sums.y;
        }
      }
    }

    // Store the result for other threads.
    if( threadIdx.x == 0 ) {
      smem_sums = sums;
    }

    // Make sure the sums are in shared memory.
    __syncthreads();

    // Read the 1st mean from shared memory.
    mean_1 = smem_sums.x;
    // Read the 2nd mean from shared memory.
    mean_2 = smem_sums.y;

    mean_1 *= params.inv_hwc_per_group;
    mean_2 *= params.inv_hwc_per_group;
    
    // The pointer to the first activation stored by that thread.
    IOType *dx_ptr = &reinterpret_cast<IOType*>(params.dx)[offset];

    // Iterate over the activations to normalize the activations and store the results.
    for( int ii = 0; ii < ACTS_PER_THREAD; ++ii ) {
      // Convert x to float.
      float2 x_f2  = IOTraits::unpack(x [ii]);
      // Convert dY to float.
      float2 dy_f2 = IOTraits::unpack(dy[ii]);

      // X - X_mean.
      float2 x_minus_x_mean_f2;
      x_minus_x_mean_f2.x = x_f2.x - x_mean; 
      x_minus_x_mean_f2.y = x_f2.y - x_mean; 
      // Normalize X.
      float2 x_norm_f2;
      x_norm_f2.x = x_minus_x_mean_f2.x * rcp_x_stddev;
      x_norm_f2.y = x_minus_x_mean_f2.y * rcp_x_stddev;

      if (params.with_swish) {
        float x_gn_x = x_norm_f2.x * gamma_f2.x + beta_f2.x;
        float x_gn_y = x_norm_f2.y * gamma_f2.y + beta_f2.y;
        float s_x = sigmoid(x_gn_x);
        float s_y = sigmoid(x_gn_y);
        dy_f2.x = dy_f2.x * s_x * (1.f + x_gn_x * (1.f - s_x));
        dy_f2.y = dy_f2.y * s_y * (1.f + x_gn_y * (1.f - s_y));
      }

      // The gradient that enters the x_norm node.
      float2 dx_norm; 
      dx_norm.x = dy_f2.x * gamma_f2.x;
      dx_norm.y = dy_f2.y * gamma_f2.y;

      // The gradient along the input path.
      float2 dx;
      dx.x = (dx_norm.x - (x_norm_f2.x * mean_1 + mean_2)) * rcp_x_stddev;
      dx.y = (dx_norm.y - (x_norm_f2.y * mean_1 + mean_2)) * rcp_x_stddev;

      // Store the scaled values.
      int hwj = hwi + ii*ACTS_PER_LOOP;
      if( is_active && hwj < params.hw ) {
        *reinterpret_cast<IOType2*>(&dx_ptr[hwj*params.c]) = IOTraits::pack(dx);
      }
    }
  }

  // The completion barrier.
  int *barrier = &params.barriers[gridDim.x == 1 ? 0 : gridDim.y*2];

  // Mark the completion of the threadblock.
  if( threadIdx.x == 0 ) {
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(barrier));
  }

  // Exit if that's not the last thread block.
  if( blockIdx.x != gridDim.x-1 || blockIdx.y != gridDim.y-1 ) {
    return;
  }

  // Busy wait. We could use found = old + step with old = atomicAdd(...) but it's not faster.
  if( threadIdx.x == 0 ) {
    for( int found = -1; found != gridDim.x * gridDim.y; ) {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(found) : "l"(barrier));
    }
  }
  __syncthreads();

  // The last block converts dgamma and dbeta to half.
  for( int idx = threadIdx.x; idx < params.c/2; idx += THREADS_PER_BLOCK ) {
    // Load dgamma.
    float2 dgamma;
    dgamma.x = params.zeroed_red_buffer[idx + 0*params.c/2];
    dgamma.y = params.zeroed_red_buffer[idx + 1*params.c/2];

    // Load dbeta.
    float2 dbeta;
    dbeta.x = params.zeroed_red_buffer[idx + 2*params.c/2];
    dbeta.y = params.zeroed_red_buffer[idx + 3*params.c/2];

    // Store to global memory.
    *reinterpret_cast<WType2*>(&reinterpret_cast<WType*>(params.dgamma)[idx*2]) = WTraits::pack(dgamma);
    *reinterpret_cast<WType2*>(&reinterpret_cast<WType*>(params.dbeta )[idx*2]) = WTraits::pack(dbeta);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
