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

#include "fmha.h"
#include "fmha_dgrad_kernel_1xN_reload.h"

using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u>;

extern "C" __global__ void fmha_dgrad_fp16_256_64_sm80_kernel(Fused_multihead_attention_fprop_params params) {
    fmha::compute_dv_1xN<Kernel_traits>(params);
    fmha::compute_dq_dk_1xN<Kernel_traits>(params);
}

void run_fmha_dgrad_fp16_256_64_sm80(const Fused_multihead_attention_fprop_params &params, cudaStream_t stream) {

    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_o = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

    using Smem_tile_s = fmha::Smem_tile_mma_transposed< Kernel_traits::Cta_tile_p>;
    constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;
    static_assert(smem_size_s == 16 * 256 * 2);
    static_assert(smem_size_o == 16 * 64 * 4 * Kernel_traits::Cta_tile_p::WARPS_N);

    constexpr int smem_size_dv = smem_size_s + 2 * smem_size_q + smem_size_v + smem_size_softmax;
    constexpr int smem_size_dq_dk = smem_size_s + smem_size_o + smem_size_q + smem_size_v;
    constexpr int smem_size = std::max(smem_size_dv, smem_size_dq_dk);

    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(
            fmha_dgrad_fp16_256_64_sm80_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 grid(params.h, params.b);
    fmha_dgrad_fp16_256_64_sm80_kernel<<<grid, Kernel_traits::THREADS, smem_size, stream>>>(params);
}
