/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <ATen/Dispatch.h>

constexpr int block_size = 512;
constexpr int ctas_per_sm = 4;

template <typename scalar_t>
__global__ void
__launch_bounds__(block_size)
mha_fill_kernel(scalar_t* out_tensor,
                const int32_t* const start_row,
                const size_t num_rows) {
    size_t row_stride = gridDim.y * blockDim.x;
    size_t row_index = blockIdx.x + (size_t)start_row[0];
    size_t col_index = blockIdx.y * blockDim.x + threadIdx.x;
    while (row_index < num_rows) {
        out_tensor[row_index*row_stride + col_index] = 0;
        row_index += gridDim.x;
    }
}

at::Tensor & mha_fill(at::Tensor &self, const at::Tensor &start_index) {
    auto max_tokens = self.size(0);
    auto self_2d = self.view({max_tokens, -1});
    auto fcd_size = self_2d.size(1);
    TORCH_CHECK (self.is_contiguous(), "input not contiguous");
    TORCH_CHECK (fcd_size % block_size == 0, "input size not aligned to block size");
    const int num_mp = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    uint64_t num_blk_y = (uint64_t)(fcd_size / block_size);
    uint64_t num_blk_x = (uint64_t)std::ceil(num_mp * ctas_per_sm / num_blk_y);
    dim3 dim_grid(num_blk_x, num_blk_y);
    dim3 dim_block(block_size);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, self_2d.scalar_type(), "mha_padding_fill_", [&]() {
            mha_fill_kernel<<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                self_2d.data_ptr<scalar_t>(), start_index.data_ptr<int32_t>(), max_tokens);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return self;
}
