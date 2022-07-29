/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <fmha_utils.h>


constexpr int TOTAL_DIM = 0;
constexpr int THREE_DIM = 1;
constexpr int H_DIM = 2;
constexpr int D_DIM = 3;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    // The QKV matrices.
    void * __restrict__ qkv_ptr;

    // The stride between rows of the Q, K and V matrices.
    size_t qkv_stride_in_bytes;

    // The number of heads.
    int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_fprop_params : public Qkv_params {

    // The dQKV matrices.
    void * __restrict__ dqkv_ptr;

    // Temporary for dKV.
    void * __restrict__ dkv_ptr;

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    int64_t o_stride_in_bytes;

    // The pointer to the S matrix, overwritten by the dP matrix (bwd).
    void * __restrict__ s_ptr;
    // The stride between rows of the S matrix.
    int64_t s_stride_in_bytes;

    // The dimensions.
    int b, s, d;

    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;

    // Scale factor of 1 / (1 - p_dropout), in half2.
    uint32_t scale_dropout;

    // Random state.
    at::PhiloxCudaState philox_args;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_params> 
struct Launch_params{
    Launch_params(cudaDeviceProp * props_,
                  cudaStream_t stream_,
                  bool is_training_,
                  bool is_nl_) 
        : elts_per_thread(0)
        , props(props_)
        , stream(stream_)
        , is_training(is_training_)
        , is_nl(is_nl_) {
    }

    size_t elts_per_thread;

    cudaDeviceProp * props;

    cudaStream_t stream;

    bool is_training;

    Kernel_params params;
    int num_full_heads;
    int num_main_groups;
    int heads_last_wave;
    int main_steps;
    int rest_steps;
    bool is_nl;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_fmha_fp16_128_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool configure);
void run_fmha_fp16_256_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool configure);
void run_fmha_fp16_384_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool configure);
void run_fmha_fp16_512_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool configure);

void run_fmha_dgrad_fp16_128_64_sm80(const Fused_multihead_attention_fprop_params &params, cudaStream_t stream);
void run_fmha_dgrad_fp16_256_64_sm80(const Fused_multihead_attention_fprop_params &params, cudaStream_t stream);
void run_fmha_dgrad_fp16_384_64_sm80(const Fused_multihead_attention_fprop_params &params, cudaStream_t stream);
void run_fmha_dgrad_fp16_512_64_sm80(const Fused_multihead_attention_fprop_params &params, cudaStream_t stream);

void run_fmha_fp16_512_64_sm80_nl(const Fused_multihead_attention_fprop_params &params, const bool is_training, const int num_chunks, cudaStream_t stream); 

void run_fmha_dgrad_fp16_512_64_sm80_nl(const Fused_multihead_attention_fprop_params &params, const int num_chunks, cudaStream_t stream);

void fmha_run_noloop_reduce(void *out,
                            const void *in,
                            const int *cu_seqlens,
                            const int hidden_size,
                            const int batch_size,
                            const int total,
                            const int num_chunks,
                            cudaStream_t stream);


