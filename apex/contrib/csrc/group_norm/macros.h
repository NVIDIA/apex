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
 
#define GN_ONE_PASS_RUN_FUNCTION_NAME(Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
void group_norm_nhwc_ ## PASS_NAME ## _one_pass_ ## CHANNELS_PER_GROUP ## _ ## ACTS_PER_BLOCK ## _ ## Traits ## _run( \
    const Group_norm_nhwc_ ## PASS_NAME ## _params &params, \
    const dim3 &grid, \
    cudaStream_t stream)

#define GN_ONE_PASS_RUN_FUNCTION(Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
GN_ONE_PASS_RUN_FUNCTION_NAME(Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) { \
  \
  auto kernel = group_norm_nhwc_ ## PASS_NAME ## _one_pass_kernel<Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK>; \
  \
  const Group_norm_nhwc_ ## PASS_NAME ## _params *params_ = &params;  \
  if( grid.x > 1 ) {  \
    CHECK_CUDA(cudaLaunchCooperativeKernel((const void*) kernel,  \
                                           grid,  \
                                           dim3(THREADS_PER_BLOCK), \
                                           (void**) &params_, \
                                           0, \
                                           stream));  \
  \
  } else {  \
    CHECK_CUDA(cudaLaunchKernel((const void*) kernel, \
                                grid, \
                                dim3(THREADS_PER_BLOCK),  \
                                (void**) &params_,  \
                                0,  \
                                stream)); \
  \
  } \
  \
  CHECK_CUDA(cudaGetLastError()); \
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#define GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION_NAME(Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
int group_norm_nhwc_ ## PASS_NAME ## _one_pass_ ## CHANNELS_PER_GROUP ## _ ## ACTS_PER_BLOCK ## _ ## Traits ## _blocks_per_sm()

#define GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION(Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION_NAME(Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) { \
  \
  auto kernel = group_norm_nhwc_ ## PASS_NAME ## _one_pass_kernel<Traits, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, THREADS_PER_BLOCK>; \
  \
  int blocks_per_sm = 0; \
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, \
                                                           kernel, \
                                                           THREADS_PER_BLOCK, \
                                                           0)); \
  \
  CHECK_CUDA(cudaGetLastError()); \
  return blocks_per_sm; \
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#define GN_ONE_PASS_(FUNCTION, Traits, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
  FUNCTION(Traits, 512,    CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  FUNCTION(Traits, 256,    CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  FUNCTION(Traits, 128,    CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  FUNCTION(Traits, 64,     CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME);

#define GN_ONE_PASS_RUN_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
  GN_ONE_PASS_(GN_ONE_PASS_RUN_FUNCTION, Fp32, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_RUN_FUNCTION, Bf16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_RUN_FUNCTION, Fp16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME);

#define GN_ONE_PASS_RUN_DECLARATION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
  GN_ONE_PASS_(GN_ONE_PASS_RUN_FUNCTION_NAME, Fp32, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_RUN_FUNCTION_NAME, Bf16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_RUN_FUNCTION_NAME, Fp16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME);

#define GN_ONE_PASS_BLOCKS_PER_SM_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
  GN_ONE_PASS_(GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION, Fp32, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION, Bf16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION, Fp16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME);

#define GN_ONE_PASS_BLOCKS_PER_SM_DECLARATION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
  GN_ONE_PASS_(GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION_NAME, Fp32, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION_NAME, Bf16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME); \
  GN_ONE_PASS_(GN_ONE_PASS_BLOCKS_PER_SM_FUNCTION_NAME, Fp16, CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME);

#define GN_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
GN_ONE_PASS_RUN_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME) \
GN_ONE_PASS_BLOCKS_PER_SM_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, PASS_NAME)

#define GN_FWD_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK) \
GN_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, fwd)

#define GN_BWD_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK) \
GN_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK, bwd)

#define GN_FWD_BWD_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK) \
GN_FWD_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK) \
GN_BWD_ONE_PASS_DEFINITION(CHANNELS_PER_GROUP, THREADS_PER_BLOCK)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define GN_SELECTION_STATEMENT(function, Traits, PRECISION, FUNC_POSTFIX, HW_THRESHOLD, ACTS_PER_BLOCK, CHANNELS_PER_GROUP, PASS_NAME) \
  if( params.hw >= HW_THRESHOLD && params.channels_per_group == CHANNELS_PER_GROUP && params.precision == PrecisionMode::PRECISION ) { \
    function = group_norm_nhwc_ ## PASS_NAME ## _one_pass_ ## CHANNELS_PER_GROUP ## _ ## ACTS_PER_BLOCK ## _ ## Traits ## FUNC_POSTFIX ; \
  } else

#define GN_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(Traits, PRECISION, CHANNELS_PER_GROUP, FUNC_POSTFIX, function, PASS_NAME) \
  GN_SELECTION_STATEMENT(function, Traits, PRECISION, FUNC_POSTFIX, 512,      512,    CHANNELS_PER_GROUP, PASS_NAME) \
  GN_SELECTION_STATEMENT(function, Traits, PRECISION, FUNC_POSTFIX, 256,      256,    CHANNELS_PER_GROUP, PASS_NAME) \
  GN_SELECTION_STATEMENT(function, Traits, PRECISION, FUNC_POSTFIX, 128,      128,    CHANNELS_PER_GROUP, PASS_NAME) \
  GN_SELECTION_STATEMENT(function, Traits, PRECISION, FUNC_POSTFIX, 0,        64,     CHANNELS_PER_GROUP, PASS_NAME)

#define GN_FWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(CHANNELS_PER_GROUP, FUNC_POSTFIX, function) \
GN_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(Fp32, FP32, CHANNELS_PER_GROUP, FUNC_POSTFIX, function, fwd) \
GN_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(Fp16, FP16, CHANNELS_PER_GROUP, FUNC_POSTFIX, function, fwd) \
GN_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(Bf16, BF16, CHANNELS_PER_GROUP, FUNC_POSTFIX, function, fwd)

#define GN_BWD_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(CHANNELS_PER_GROUP, FUNC_POSTFIX, function) \
GN_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(Fp32, FP32, CHANNELS_PER_GROUP, FUNC_POSTFIX, function, bwd) \
GN_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(Fp16, FP16, CHANNELS_PER_GROUP, FUNC_POSTFIX, function, bwd) \
GN_SELECTION_STATEMENT_HW_THRESHOLD_ACTS_PER_BLOCK_DISPATCH(Bf16, BF16, CHANNELS_PER_GROUP, FUNC_POSTFIX, function, bwd)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define GN_ONE_PASS_DECLARATION(CHANNELS_PER_GROUP, PASS_NAME) \
GN_ONE_PASS_RUN_DECLARATION(CHANNELS_PER_GROUP, /* dummy value */ 0, PASS_NAME) \
GN_ONE_PASS_BLOCKS_PER_SM_DECLARATION(CHANNELS_PER_GROUP, /* dummy value */ 0, PASS_NAME)

#define GN_FWD_ONE_PASS_DECLARATION(CHANNELS_PER_GROUP) \
GN_ONE_PASS_DECLARATION(CHANNELS_PER_GROUP, fwd)

#define GN_BWD_ONE_PASS_DECLARATION(CHANNELS_PER_GROUP) \
GN_ONE_PASS_DECLARATION(CHANNELS_PER_GROUP, bwd)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CALL_TWO_PASS_KERNEL(Kernel, Precision) \
  if( params.channels_per_block == 320 ) { \
    Kernel<Precision, 160><<<grid, 160, 0, stream>>>(params); \
  } else if( params.channels_per_block == 280 ) { \
    Kernel<Precision, 140><<<grid, 140, 0, stream>>>(params); \
  } else if( params.channels_per_block == 240 ) { \
    Kernel<Precision, 120><<<grid, 120, 0, stream>>>(params); \
  } else if( params.channels_per_block == 512 ) { \
    Kernel<Precision, 256><<<grid, 256, 0, stream>>>(params); \
  } else if( params.channels_per_block == 448 ) { \
    Kernel<Precision, 448><<<grid, 224, 0, stream>>>(params); \
  } else if( params.channels_per_block == 384 ) { \
    Kernel<Precision, 192><<<grid, 192, 0, stream>>>(params); \
  } else if( params.channels_per_block == 256 ) { \
    Kernel<Precision, 128><<<grid, 128, 0, stream>>>(params); \
  } else if( params.channels_per_block == 128 ) { \
    Kernel<Precision, 64><<<grid, 64, 0, stream>>>(params); \
  } else if( params.channels_per_block == 336 ) { \
    Kernel<Precision, 168><<<grid, 168, 0, stream>>>(params); \
  } else if( params.channels_per_block == 392 ) { \
    Kernel<Precision, 196><<<grid, 196, 0, stream>>>(params); \
  } else { \
    assert(false); \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////
