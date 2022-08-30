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
#include "fmha_fprop_kernel_1xN.h"

using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u>;

template<bool Is_training>
__global__ 
void fmha_fprop_fp16_128_64_sm80_kernel(Fused_multihead_attention_fprop_params params,
                                           const int total_heads) {

    fmha::device_1xN<Kernel_traits, Is_training>(params, total_heads);
}

void run_fmha_fp16_128_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool configure) {

    auto kernel = launch_params.is_training ? &fmha_fprop_fp16_128_64_sm80_kernel<true> : &fmha_fprop_fp16_128_64_sm80_kernel<false>;

    constexpr int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>();

    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    const int sm_count = launch_params.props->multiProcessorCount;
    int ctas_per_sm;
    FMHA_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size));
    int total_ctas = sm_count * ctas_per_sm;

    const int heads_total = launch_params.params.b * launch_params.params.h;
    if(configure) {

        using Mma_tile_p = fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>;
        constexpr size_t STEPS = Kernel_traits::Cta_tile_p::N / Kernel_traits::Cta_tile_p::M;
        constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
        constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;

        size_t heads_per_cta = ((heads_total + total_ctas - 1) / total_ctas);
        size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8;
        launch_params.elts_per_thread = heads_per_cta * elts_per_head;
        return;
    }

    dim3 grid(total_ctas);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
        launch_params.params,
        heads_total);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());

}


