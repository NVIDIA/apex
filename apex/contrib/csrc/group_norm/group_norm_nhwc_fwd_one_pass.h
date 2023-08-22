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
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// F O R W A R D
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#define GN_FWD_SELECT(FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(4, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(8, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(10, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(14, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(16, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(20, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(26, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(24, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(28, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(30, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(32, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(40, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(42, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(48, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(56, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(60, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(64, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(70, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(80, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(84, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(96, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(98, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(112, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(120, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(128, FUNC_POSTFIX, function) \
  GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(160, FUNC_POSTFIX, function) \
  { \
    assert(false && "Not implemented"); \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

#define GN_FWD_RUNNER_SELECT(function) \
  GN_FWD_SELECT(_run, function)

#define GN_FWD_BLOCKS_PER_SM_SELECT(function) \
  GN_FWD_SELECT(_blocks_per_sm, function)

////////////////////////////////////////////////////////////////////////////////////////////////////

GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */   4)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */   8)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  10)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  14)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  16)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  20)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  26)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  24)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  28)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  30)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  32)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  40)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  42)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  48)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  56)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  60)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  64)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  70)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  80)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  84)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  96)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */  98)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */ 112)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */ 120)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */ 128)
GN_FWD_ONE_PASS_DECLARATION(/* CHANNELS_PER_GROUP */ 160)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline void group_norm_nhwc_fwd_one_pass_setup(Group_norm_nhwc_fwd_params &params, 
                                        size_t &barriers_elts,
                                        size_t &red_buffer_elts,
                                        dim3 &grid,
                                        const cudaDeviceProp &props) {

  // The pre-computed dimensions.
  params.hw  = params.h * params.w;
  params.hwc = params.c * params.hw;

  // The number of channels per group.
  params.channels_per_group = params.c / params.groups;
  // The inverse to compute the mean/variance.
  params.inv_hwc_per_group = 1.f / (float) (params.hw * params.channels_per_group);

  // Select the kernel.
  using Function_t = int (*)(); 

  Function_t blocks_per_sm_function;
  GN_FWD_BLOCKS_PER_SM_SELECT(blocks_per_sm_function);

  // Define how many activations are computed per block.
  if( params.hw >= 1024 && params.channels_per_group >= 80 ||
      (params.hw >= 256 && params.channels_per_group >= 160) ) 
  {
    params.acts_per_block = 8 * 16;
  } else if( params.hw >= 512 ) {
    params.acts_per_block = 16 * 32;
  } else if( params.hw >= 256 ) {
    params.acts_per_block = 16 * 16;
  } else if( params.hw >= 128 ) {
    params.acts_per_block = 8 * 16;
  } else if ( params.hw > 0 ) {
    params.acts_per_block = 8 * 8;
  } else {
    // We should never be here if params are set correctly.
    assert(false);
  }

  // Define the number of blocks per activation map. TODO: Make sure it matches the kernel sizes.
  int blocks_per_slice = div_up(params.hw, params.acts_per_block);

  // The number of blocks that can be run per SM.
  int blocks_per_sm = blocks_per_sm_function();

  // The number of blocks per grid.
  int max_blocks_per_grid = blocks_per_sm * props.multiProcessorCount;

  // Make sure we are safe to run that many blocks
  assert(blocks_per_slice <= max_blocks_per_grid);

  // The number of blocks per slice is the X dimension of the grid.
  grid.x = blocks_per_slice;
  // The number of groups *  is the X dimension of the grid.
  grid.y = std::min(max_blocks_per_grid / blocks_per_slice, params.groups * params.n);

  // The number of barriers.
  barriers_elts = blocks_per_slice > 1 ? grid.y * 2 : 0;

  // The number of elements in the reduction buffer (for the sums and sums of squared). 
  if( blocks_per_slice == 1 ) {
    red_buffer_elts = 0;
  } else {
    // The first 2 is for double-buffering. The 2nd one is for the fact that we have two floats.
    red_buffer_elts = 2 * grid.x * grid.y * 2;
  }
}

inline void group_norm_nhwc_fwd_one_pass_run(const Group_norm_nhwc_fwd_params &params, 
                                      const dim3 &grid, 
                                      cudaStream_t stream) {

  using Function_t = void (*)(const Group_norm_nhwc_fwd_params &, 
                              const dim3 &, 
                              cudaStream_t); 

  Function_t runner;
  GN_FWD_RUNNER_SELECT(runner);

  runner(params, grid, stream);
}
