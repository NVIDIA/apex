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

using Kernel_traits = FMHA_kernel_traits<384, 64, 16, 1, 4, 0x18u>;

template<bool Is_training>
__global__ 
void fmha_fprop_fp16_384_64_sm80_kernel(Fused_multihead_attention_fprop_params params,
                                           const int num_full_heads,
                                           const int num_main_groups,
                                           const int main_group_size,
                                           const int main_steps,
                                           const int rest_steps) {

    fmha::device_1xN<Kernel_traits, Is_training>(
        params, num_full_heads, num_main_groups, main_group_size, main_steps, rest_steps);
}

void run_fmha_fp16_384_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool configure) {

    auto kernel = launch_params.is_training ? &fmha_fprop_fp16_384_64_sm80_kernel<true> : &fmha_fprop_fp16_384_64_sm80_kernel<false>;

    constexpr int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>();

    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    const int sm_count = launch_params.props->multiProcessorCount;
    int ctas_per_sm;
    FMHA_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size));
    int total_ctas = sm_count * ctas_per_sm;

    if(configure) {
        const int heads_total = launch_params.params.b * launch_params.params.h;
        std::tie(launch_params.num_full_heads,
                 launch_params.num_main_groups, 
                 launch_params.heads_last_wave, 
                 launch_params.main_steps, 
                 launch_params.rest_steps, 
                 launch_params.elts_per_thread) = fmha::work_dist<Kernel_traits>(total_ctas, heads_total);
        return;
    }

    dim3 grid(total_ctas);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
        launch_params.params,
        launch_params.num_full_heads, 
        launch_params.num_main_groups, 
        launch_params.heads_last_wave, 
        launch_params.main_steps, 
        launch_params.rest_steps);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());

}

