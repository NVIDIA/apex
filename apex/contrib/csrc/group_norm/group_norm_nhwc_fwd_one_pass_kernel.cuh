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
// F O R W A R D
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, int ACTS_PER_BLOCK_, int CHANNELS_PER_GROUP_, int THREADS_PER_BLOCK_ >
__global__ __launch_bounds__(THREADS_PER_BLOCK_) 
  void group_norm_nhwc_fwd_one_pass_kernel(Group_norm_nhwc_fwd_params params) {

  // The traits.
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

    // The offset to the first activation loaded by that thread.
    const int64_t offset = (int64_t) ni*params.hwc + gi*CHANNELS_PER_GROUP + ci;
    // The pointer to the first activation loaded by that thread.
    const IOType *x_ptr = &reinterpret_cast<const IOType*>(params.x)[offset];

    // Load the activations into registers.
    IOType2 x[ACTS_PER_THREAD];
    #pragma unroll
    for( int ii = 0; ii < ACTS_PER_THREAD; ++ii ) {
      int hwj = hwi + ii*ACTS_PER_LOOP;
      x[ii] = IOTraits::zero();
      if( is_active && hwj < params.hw ) {
        x[ii] = *reinterpret_cast<const IOType2*>(&x_ptr[hwj*params.c]);
      }
    }

    // Compute the sum and the sum of squares for each thread.
    float2 sums = make_float2(0.f, 0.f);
    #pragma unroll
    for( int ii = 0; ii < ACTS_PER_THREAD; ++ii ) {
      float2 f2 = IOTraits::unpack(x[ii]);
      sums.x += f2.x + f2.y;
      sums.y += f2.x * f2.x + f2.y * f2.y;
    }

    // Clear invalid threads.
    if( ACTIVE_THREADS < THREADS_PER_BLOCK && !is_active ) {
      sums = make_float2(0.f, 0.f);
    }

    // Compute the sums for the block.
    sums = Block_reduce(temp_storage).Reduce(sums, [](const float2 &a, const float2 &b) {
      return make_float2(a.x + b.x, a.y + b.y);
    });

    // The block leader stores to global memory, if needed.
    if( gridDim.x > 1 ) {

      // The index of the buffer (double-buffering).
      int red_buffer_idx = step & 1;
      // The barrier.
      int *barrier = &params.barriers[red_buffer_idx*gridDim.y + blockIdx.y];
      // The offset to the reduction buffer.
      int red_buffer_offset = red_buffer_idx*gridDim.x*gridDim.y*2;
      // The reduction buffer.
      float2 *red_buffer = reinterpret_cast<float2*>(&params.red_buffer[red_buffer_offset]);

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

    // Store the results to global memory as well (for training).
    if( params.sums != nullptr && blockIdx.x == 0 && threadIdx.x == 0 ) {
      sums.x *= params.inv_hwc_per_group;
      sums.y *= params.inv_hwc_per_group;
      params.sums[ngi] = sums;
    }

    // Make sure the sums are in shared memory.
    __syncthreads();

    // Load gamma/beta.
    float2 gamma_f2 = WTraits::unpack(*reinterpret_cast<const WType2*>(
      &reinterpret_cast<const WType*>(params.gamma)[gi*CHANNELS_PER_GROUP+ci]));
    float2 beta_f2  = WTraits::unpack(*reinterpret_cast<const WType2*>(
      &reinterpret_cast<const WType*>(params.beta) [gi*CHANNELS_PER_GROUP+ci]));

    // Compute the mean.
    float mean = smem_sums.x * params.inv_hwc_per_group;
    // Compute the variance.
    float var = smem_sums.y * params.inv_hwc_per_group - (mean * mean);
    // Compute the inverse of the stddev.
    float inv_stddev = var <= 0.f ? 1.f : rsqrtf(var + params.epsilon);

    // The pointer to the first activation stored by that thread.
    IOType *y_ptr = &reinterpret_cast<IOType*>(params.y)[offset];

    // Iterate over the activations to normalize the activations and store the results.
    for( int ii = 0; ii < ACTS_PER_THREAD; ++ii ) {

      // Extract the two half values.
      float2 f2 = IOTraits::unpack(x[ii]);

      // Normalize the channels.
      f2.x = (f2.x - mean) * inv_stddev;
      f2.y = (f2.y - mean) * inv_stddev;

      // Scale by gamma and add beta.
      f2.x = gamma_f2.x * f2.x + beta_f2.x;
      f2.y = gamma_f2.y * f2.y + beta_f2.y;

      // Apply Swish if needed.
      if( params.with_swish ) {
        f2.x = f2.x * sigmoid(f2.x);
        f2.y = f2.y * sigmoid(f2.y);
      }

      // Store the scaled values.
      int hwj = hwi + ii*ACTS_PER_LOOP;
      if( is_active && hwj < params.hw ) {
        *reinterpret_cast<IOType2*>(&y_ptr[hwj*params.c]) = IOTraits::pack(f2);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
