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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "fmha.h"

void run_fmha_fp16_128_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                  bool is_training,
                                  cudaStream_t stream);
void run_fmha_fp16_256_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                  bool is_training,
                                  cudaStream_t stream);
void run_fmha_fp16_384_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                  bool is_training,
                                  cudaStream_t stream);
void run_fmha_fp16_512_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                  bool is_training,
                                  cudaStream_t stream);

void run_fmha_dgrad_fp16_128_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                        cudaStream_t stream);
void run_fmha_dgrad_fp16_256_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                        cudaStream_t stream);
void run_fmha_dgrad_fp16_384_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                        cudaStream_t stream);
void run_fmha_dgrad_fp16_512_64_sm80(const Fused_multihead_attention_fprop_params &params,
                                        cudaStream_t stream);

void set_params(Fused_multihead_attention_fprop_params &params,
                // sizes
                const size_t b,
                const size_t s,
                const size_t h,
                const size_t d,
                // device pointers
                void *qkv_packed_d,
                void *cu_seqlens_d,
                void *seqlens_d,
                void *o_packed_d,
                void *s_d,
                float p_dropout) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.qkv_ptr = qkv_packed_d;
    params.qkv_stride_in_bytes = get_size_in_bytes(h * 3 * d, data_type);
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);
    params.seqlens = static_cast<int *>(seqlens_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    const float scale_bmm1 = 1.f / sqrtf(d);
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;

    set_alpha(params.scale_bmm1, scale_bmm1, acc_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    params.rp_dropout = 1.f / params.p_dropout;
    TORCH_CHECK(p_dropout < 1.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);
}

constexpr uint32_t NUM_HEADS_DIM = 2;
constexpr uint32_t THREE_DIM = 1;

std::vector<at::Tensor>
mha_fwd(const at::Tensor &qkv,  // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
        const at::Tensor &cu_seqlens,  // b+1
        const at::Tensor &seqlens,     // b
        const float p_dropout,
        const int max_seq_len,
        const bool is_training,
        c10::optional<at::Generator> gen_) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor == 0);
    int seq_len = 512;
    auto launch = &run_fmha_fp16_512_64_sm80;
    if( max_seq_len <= 128 ) {
        seq_len = 128;
        launch = &run_fmha_fp16_128_64_sm80;
    } else if( max_seq_len <= 256 ) {
        seq_len = 256;
        launch = &run_fmha_fp16_256_64_sm80;
    } else if( max_seq_len <= 384 ) {
        seq_len = 384;
        launch = &run_fmha_fp16_384_64_sm80;
    } else if( max_seq_len <= 512 ) {
        seq_len = 512;
        launch = &run_fmha_fp16_512_64_sm80;
    } else {
        TORCH_CHECK(false);
    }

    constexpr int warps_m = 1;
    constexpr int warps_n = 4;  // this leads to an upper bound
    const int mmas_m = seq_len / 16 / warps_m;
    const int mmas_n = seq_len / 16 / warps_n;
    
    const int elts_per_thread = 8 * mmas_m * mmas_n;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(qkv.dtype() == torch::kFloat16);
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32);
    TORCH_CHECK(seqlens.dtype() == torch::kInt32);

    TORCH_CHECK(qkv.is_cuda())
    TORCH_CHECK(cu_seqlens.is_cuda())

    TORCH_CHECK(qkv.is_contiguous())
    TORCH_CHECK(cu_seqlens.is_contiguous())
    TORCH_CHECK(seqlens.is_contiguous())

    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);

    const auto sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    const int batch_size = cu_seqlens.numel() - 1;
    TORCH_CHECK(seqlens.numel() == batch_size);
    const int total = sizes[0];
    const int num_heads = sizes[NUM_HEADS_DIM];
    const int head_size = sizes[3];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 64);
    auto opts = qkv.options();

    auto ctx = torch::empty({ total, num_heads, head_size }, opts);

    auto s = torch::empty({ batch_size, num_heads, seq_len, seq_len }, opts);

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               qkv.data_ptr(),
               cu_seqlens.data_ptr(),
               seqlens.data_ptr(),
               ctx.data_ptr(),
               s.data_ptr(),
               p_dropout);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = elts_per_thread;
    at::PhiloxCudaState rng_engine_inputs;

    if( is_training ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    launch(params, is_training, stream);

    return { ctx, s };
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // total x num_heads, x head_size
        const at::Tensor &qkv,   // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
        at::Tensor &softmax,     // b x h x s x s softmax and dmask - will be overwritten with dP
        const at::Tensor &cu_seqlens,  // b+1
        const at::Tensor &seqlens,     // b
        const float p_dropout,         // probability to drop
        const int max_seq_len          // max sequence length to choose the kernel
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 8 && dprops->minor == 0);
    int seq_len = 512;
    auto launch = &run_fmha_dgrad_fp16_512_64_sm80;
    if( max_seq_len <= 128 ) {
        seq_len = 128;
        launch = &run_fmha_dgrad_fp16_128_64_sm80;
    } else if( max_seq_len <= 256 ) {
        seq_len = 256;
        launch = &run_fmha_dgrad_fp16_256_64_sm80;
    } else if( max_seq_len <= 384 ) {
        seq_len = 384;
        launch = &run_fmha_dgrad_fp16_384_64_sm80;
    } else if( max_seq_len <= 512 ) {
        seq_len = 512;
        launch = &run_fmha_dgrad_fp16_512_64_sm80;
    } else {
        TORCH_CHECK(false);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(qkv.dtype() == torch::kFloat16);
    TORCH_CHECK(dout.dtype() == torch::kFloat16);
    TORCH_CHECK(softmax.dtype() == torch::kFloat16);
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32);
    TORCH_CHECK(seqlens.dtype() == torch::kInt32);

    TORCH_CHECK(qkv.is_cuda());
    TORCH_CHECK(cu_seqlens.is_cuda());

    TORCH_CHECK(qkv.is_contiguous());
    TORCH_CHECK(cu_seqlens.is_contiguous());
    TORCH_CHECK(seqlens.is_contiguous());

    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);

    const auto sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    const int batch_size = cu_seqlens.numel() - 1;
    TORCH_CHECK(seqlens.numel() == batch_size);
    const int num_heads = sizes[NUM_HEADS_DIM];
    const int head_size = sizes[3];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 64);

    auto dqkv = torch::empty_like(qkv);

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               qkv.data_ptr(),
               cu_seqlens.data_ptr(),
               seqlens.data_ptr(),
               dout.data_ptr(),     // we set o_ptr to dout
               softmax.data_ptr(),  // softmax gets overwritten by dP!
               p_dropout);

    // we're re-using these scales scales
    Data_type acc_type = DATA_TYPE_FP32;
    set_alpha(params.scale_bmm1, 1.f, acc_type);
    set_alpha(params.scale_softmax, 1.f / sqrtf(head_size), acc_type);
    set_alpha(params.scale_bmm2, 1.f, DATA_TYPE_FP16);
    params.dqkv_ptr = dqkv.data_ptr();

    launch(params, stream);
    return { dqkv, softmax };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused Multi-head Self-attention for BERT";  
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
}
