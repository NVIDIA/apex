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
#include "macros.h"
#include "traits.h"
#include <assert.h>
#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// B A C K W A R D
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, int THREADS_PER_BLOCK >
__global__ void group_norm_nhwc_bwd_sum_kernel(Group_norm_nhwc_bwd_params params) {

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

  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<Group_sums, THREADS_PER_BLOCK> Block_scan;

  // Allocate shared memory for Block_scan.
  __shared__ typename Block_scan::TempStorage temp_storage;
  // Allocate shared memory for the groups. We could reduce the amount of shared memory reserved.
  __shared__ float2 smem[THREADS_PER_BLOCK];

  // The instance in the batch.
  int ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int ci = blockIdx.x * params.channels_per_block + threadIdx.x * 2;
  // The group that thread works on and the channel in the group (modulus).
  int gi = ci / params.channels_per_group;

  // The sums from the fwd pass.
  float2 fwd = params.sums[ni*params.groups + gi];
  // The mean of X (computed during the fwd pass -- one value per batch*group).
  float x_mean = fwd.x;
  // The mean of squares of X (computed during the fwd pass -- one value per batch*group).
  float x_sq_mean = fwd.y;
  // The variance.
  float x_var = x_sq_mean - x_mean * x_mean;
  // The reciprocal of the standard deviation (i.e. 1.f / sqrt(var + epsilon)).
  float rcp_x_stddev = x_var <= 0.f ? 1.f : 1.f / sqrtf(x_var + params.epsilon);

  // Load gamma.
  float2 gamma_f2 = make_float2(0.f, 0.f);
  float2 beta_f2 = make_float2(0.f, 0.f);
  if( ci < params.c ) {
    gamma_f2 = WTraits::unpack(*reinterpret_cast<const WType2*>(
      &reinterpret_cast<const WType*>(params.gamma)[ci]));
    if (params.with_swish) {
      beta_f2  = WTraits::unpack(*reinterpret_cast<const WType2*>(
        &reinterpret_cast<const WType*>(params.beta)[ci]));
    }
  }

  // The group that thread works on and the channel in the group (modulus).
  int gj = threadIdx.x * 2 / params.channels_per_group;
  int cj = threadIdx.x * 2 - params.channels_per_group * gj;

  // The first activation loaded by that block.
  int hw_begin = blockIdx.y * params.acts_per_block;
  // The last activation loaded by that block.
  int hw_end = min(hw_begin + params.acts_per_block, params.hw);

  // The gradients for gamma/beta.
  float2 dgamma = make_float2(0.f, 0.f), dbeta = make_float2(0.f, 0.f);
  // Accumulated gradients for dgrad calculation
  float mean_1 = 0.f, mean_2 = 0.f;

  // Iterate over the activations to compute the sums.
  for( int hwi = hw_begin; hwi < hw_end; ++hwi ) {

    // The offset.
    int64_t offset = (int64_t) ni*params.hwc + hwi*params.c + ci;

    // Fetch two channels per thread.
    IOType2 x_v2 = IOTraits::zero();
    IOType2 dy_v2 = IOTraits::zero();
    if( ci < params.c ) {
      x_v2  = *reinterpret_cast<const IOType2*>(&reinterpret_cast<const IOType*>(params.x )[offset]);
      dy_v2 = *reinterpret_cast<const IOType2*>(&reinterpret_cast<const IOType*>(params.dy)[offset]);
    }

    // Extract the two half values.
    float2 x_f2  = IOTraits::unpack(x_v2);
    float2 dy_f2 = IOTraits::unpack(dy_v2);

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
    dbeta.x += dy_f2.x;
    dbeta.y += dy_f2.y;

    // Update dgamma.
    dgamma.x += dy_f2.x * x_norm_x;
    dgamma.y += dy_f2.y * x_norm_y;

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

  // The data for the summations.
  Group_sums inp {cj == 0 ? 1 : 0, mean_1, mean_2};

  // Do the segmented scan.
  Group_sums out;
  Block_scan(temp_storage).InclusiveScan(inp, out, Group_sums_op());

  // Store the results for the groups in shared memory (to produce coalesced stores later).
  if( cj == params.channels_per_group - 2 /* 2 channels per thread */ ) {
    smem[gj] = make_float2(out.sum, out.sum_sq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The global group index.
  int gk = blockIdx.x * params.groups_per_block + threadIdx.x;

  // The first threads (those storing to global memory, load the values).
  float2 sums = smem[threadIdx.x];

  // Store to global memory.
  if( threadIdx.x < params.groups_per_block && gk < params.groups ) {
    atomicAdd(&params.zeroed_red_buffer[(2*ni+0)*params.groups + gk], sums.x);
    atomicAdd(&params.zeroed_red_buffer[(2*ni+1)*params.groups + gk], sums.y);
  }

  // The base pointer for the gradients for gamma and beta.
  float *dgamma_beta_ptr = &params.zeroed_red_buffer[params.n*params.groups*2];

  // The 1st channel in the output tensor. NOTE: Two channels per thread store interleaved.
  int ck = blockIdx.x * params.channels_per_block + threadIdx.x;

  // Store dgamma and dbeta as well.
  if( ck < params.c ) {
    atomicAdd(&dgamma_beta_ptr[0*params.c + 0*blockDim.x + ck], dgamma.x);
    atomicAdd(&dgamma_beta_ptr[0*params.c + 1*blockDim.x + ck], dgamma.y);
    atomicAdd(&dgamma_beta_ptr[1*params.c + 0*blockDim.x + ck], dbeta .x);
    atomicAdd(&dgamma_beta_ptr[1*params.c + 1*blockDim.x + ck], dbeta .y);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_bwd_two_passes_setup(Group_norm_nhwc_bwd_params &params,
                                          size_t &zeroed_red_buffer_elts) {

  // The pre-computed dimensions.
  params.hw  = params.h * params.w;
  params.hwc = params.c * params.hw;

  // The number of channels per group.
  params.channels_per_group = params.c / params.groups;
  // The inverse to compute the mean/variance.
  params.inv_hwc_per_group = 1.f / (float) (params.hw * params.channels_per_group);

  // Define the number of blocks per activation map. That's a simple heuristic.
  int blocks_per_act_slice = 0;
         if( params.c >= 1280 ) { 
    blocks_per_act_slice = 128 / params.n;
  } else if( params.c >= 640 ) {
    blocks_per_act_slice = 256 / params.n;
  } else {
    blocks_per_act_slice = 512 / params.n;
  }

  // Make sure we launch blocks per activation is no less than activations
  blocks_per_act_slice = min(blocks_per_act_slice, div_up(params.hw, params.n));

  // Define how many activations are computed per block.
  params.acts_per_block = div_up(params.hw, blocks_per_act_slice);

  // The number of channels per block.
  params.channels_per_block = 320;
  // Special case to deal with 30 channels per group.
  if( params.channels_per_block % params.channels_per_group != 0 ) {
    params.channels_per_block = 240;
  }

  // Special case to deal with 70 channels per group.
  if( params.c == 2240 ) {
    params.channels_per_block = 280;
  }
  else if (params.c == 832){
    params.channels_per_block = 208;
  }

  if (params.c % params.channels_per_block != 0) {
    if (params.c % 512 == 0 && params.c != 1536 && params.c != 3072 && params.c % 448 != 0) {
      params.channels_per_block = 512;
    } else if (params.c % 42 == 0) {
      params.channels_per_block = 336;
    } else if (params.c % 384 == 0) {
      params.channels_per_block = 384;
    } else if (params.c % 256 == 0 && params.c % 448 != 0 && params.c % 392 != 0) {
      params.channels_per_block = 256;
    } else if (params.c % 128 == 0 && params.c % 448 != 0 && params.c % 392 != 0) {
      params.channels_per_block = 128;
    } else if (params.c % 448 == 0 && params.c % 392 != 0) {
      params.channels_per_block = 448;
    } else if (params.c % 392 == 0) {
      params.channels_per_block = 392;
    }
  }

  // The number of groups per block.
  params.groups_per_block = params.channels_per_block / params.channels_per_group;

  // Make sure the number of channels is a multiple of the number of channels per block.
  assert(params.c % params.channels_per_block == 0);
  // Make sure a group does not span multiple blocks.
  assert(params.channels_per_block % params.channels_per_group == 0);

  // The number of elements in the reduction buffer (for the sums and sums of squared). 
  zeroed_red_buffer_elts = params.n * params.groups * 2 + params.c * 2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_bwd_two_passes_sum(const Group_norm_nhwc_bwd_params &params, 
                                        cudaStream_t stream) {

  // The dimension of the grid.
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.channels_per_block;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = div_up(params.hw, params.acts_per_block);
  // The number of instances.
  grid.z = params.n;

  if (params.precision == PrecisionMode::FP16IOFP16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Fp16IOFp16W)
  } else if (params.precision == PrecisionMode::FP16IOBF16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Fp16IOBf16W)
  } else if (params.precision == PrecisionMode::FP16IOFP32W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Fp16IOFp32W)
  } else if (params.precision == PrecisionMode::BF16IOFP16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Bf16IOFp16W)
  } else if (params.precision == PrecisionMode::BF16IOBF16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Bf16IOBf16W)
  } else if (params.precision == PrecisionMode::BF16IOFP32W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Bf16IOFp32W)
  } else if (params.precision == PrecisionMode::FP32IOFP16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Fp32IOFp16W)
  } else if (params.precision == PrecisionMode::FP32IOBF16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Fp32IOBf16W)
  } else {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_sum_kernel, Fp32IOFp32W)
  }

  // Make sure it launched ok.
  CHECK_CUDA(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, int THREADS_PER_BLOCK >
__global__ void group_norm_nhwc_bwd_scale_kernel(Group_norm_nhwc_bwd_params params) {

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

  // The instance in the batch.
  int ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int ci = blockIdx.x * params.channels_per_block + threadIdx.x * 2;
  // The group that thread works on and the channel in the group (modulus).
  int gi = ci / params.channels_per_group;

  // Load the gradients for the group.
  float mean_1 = 0.f, mean_2 = 0.f;
  if( gi < params.groups ) {
    mean_1 = params.zeroed_red_buffer[(2*ni+0)*params.groups + gi];
    mean_2 = params.zeroed_red_buffer[(2*ni+1)*params.groups + gi];
  }

  // The sums from the fwd pass.
  float2 fwd = params.sums[ni*params.groups + gi];
  // The mean of X (computed during the fwd pass -- one value per batch*group).
  float x_mean = fwd.x;
  // The mean of squares of X (computed during the fwd pass -- one value per batch*group).
  float x_sq_mean = fwd.y;
  // The variance.
  float x_var = x_sq_mean - x_mean * x_mean;
  // The reciprocal of the standard deviation (i.e. 1.f / sqrt(var + epsilon)).
  float rcp_x_stddev = x_var <= 0.f ? 1.f : 1.f / sqrtf(x_var + params.epsilon);

  // Mutiply by 1/(HWC) to get real mean
  mean_1 *= params.inv_hwc_per_group;
  mean_2 *= params.inv_hwc_per_group;

  // Load gamma.
  float2 gamma_f2 = make_float2(0.f, 0.f);
  float2 beta_f2 = make_float2(0.f, 0.f);
  if( ci < params.c ) {
    gamma_f2 = WTraits::unpack(*reinterpret_cast<const WType2*>(
      &reinterpret_cast<const WType*>(params.gamma)[ci]));
    if (params.with_swish) {
      beta_f2 = WTraits::unpack(*reinterpret_cast<const WType2*>(
        &reinterpret_cast<const WType*>(params.beta)[ci]));
    }
  }

  // The first activation loaded by that block.
  int hw_begin = blockIdx.y * params.acts_per_block;
  // The last activation loaded by that block.
  int hw_end = min(hw_begin + params.acts_per_block, params.hw);

  // Iterate over the activations to compute the sums.
  for( int hwi = hw_begin; hwi < hw_end; ++hwi ) {

    // The src/dst offset.
    int64_t offset = (int64_t) ni*params.hwc + hwi*params.c + ci;

    // Fetch two channels per thread.
    IOType2 x_v2 = IOTraits::zero();
    IOType2 dy_v2 = IOTraits::zero();
    if( ci < params.c ) {
      x_v2  = *reinterpret_cast<const IOType2*>(&reinterpret_cast<const IOType*>(params.x )[offset]);
      dy_v2 = *reinterpret_cast<const IOType2*>(&reinterpret_cast<const IOType*>(params.dy)[offset]);
    }

    // Extract the two half values.
    float2 x_f2  = IOTraits::unpack(x_v2);
    float2 dy_f2 = IOTraits::unpack(dy_v2);

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
    if( ci < params.c ) {
      *reinterpret_cast<IOType2*>(&reinterpret_cast<IOType*>(params.dx)[offset]) = IOTraits::pack(dx);
    }
  }

  // Load gamma/beta and convert to half.
  if( blockIdx.y > 0 || blockIdx.z > 0 || ci >= params.c ) {
    return;
  }

  // The base pointer for the gradients for gamma and beta.
  float *dgamma_beta_ptr = &params.zeroed_red_buffer[params.n*params.groups*2];

  // The 1st channel in the output tensor. NOTE: Two channels per thread store interleaved.
  int ck = blockIdx.x * params.channels_per_block + threadIdx.x;

  // Load the FP32 version of dgamma and dbeta. 
  float2 dgamma, dbeta;
  if (ck < params.c) {
    dgamma.x = dgamma_beta_ptr[0*params.c + 0*blockDim.x + ck];
    dgamma.y = dgamma_beta_ptr[0*params.c + 1*blockDim.x + ck];
    dbeta.x  = dgamma_beta_ptr[1*params.c + 0*blockDim.x + ck];
    dbeta.y  = dgamma_beta_ptr[1*params.c + 1*blockDim.x + ck];

    // Convert to half2 and store to memory.
    *reinterpret_cast<WType2*>(&reinterpret_cast<WType*>(params.dgamma)[ci]) = WTraits::pack(dgamma);
    *reinterpret_cast<WType2*>(&reinterpret_cast<WType*>(params.dbeta )[ci]) = WTraits::pack(dbeta);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void group_norm_nhwc_bwd_two_passes_scale(const Group_norm_nhwc_bwd_params &params, 
                                          cudaStream_t stream) {

  // The dimension of the grid.
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.channels_per_block;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = div_up(params.hw, params.acts_per_block);
  // The number of instances.
  grid.z = params.n;

  if (params.precision == PrecisionMode::FP16IOFP16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Fp16IOFp16W)
  } else if (params.precision == PrecisionMode::FP16IOBF16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Fp16IOBf16W)
  } else if (params.precision == PrecisionMode::FP16IOFP32W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Fp16IOFp32W)
  } else if (params.precision == PrecisionMode::BF16IOFP16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Bf16IOFp16W)
  } else if (params.precision == PrecisionMode::BF16IOBF16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Bf16IOBf16W)
  } else if (params.precision == PrecisionMode::BF16IOFP32W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Bf16IOFp32W)
  } else if (params.precision == PrecisionMode::FP32IOFP16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Fp32IOFp16W)
  } else if (params.precision == PrecisionMode::FP32IOBF16W) {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Fp32IOBf16W)
  } else {
    CALL_TWO_PASS_KERNEL(group_norm_nhwc_bwd_scale_kernel, Fp32IOFp32W)
  }

  // Make sure it launched ok.
  CHECK_CUDA(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

