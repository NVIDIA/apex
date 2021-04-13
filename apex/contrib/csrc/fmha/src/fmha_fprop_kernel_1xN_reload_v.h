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

#include "fmha_kernel.h"
#include <fmha/kernel_traits.h>
#include <fmha/gemm.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_training, typename Params> inline __device__ void device_1xN(const Params &params) {

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = fmha::Hmma_tile<Cta_tile_o>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    // Shared memory.
    extern __shared__ char smem_[];

    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.x;
    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early() )
        return;

    Mask<Cta_tile_p> mask(params, binfo, tidx);

    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph(std::get<0>(seeds), binfo.tidx_global, std::get<1>(seeds));

    static_assert(2 * Mma_tile_p::MMAS_M * 4 * Mma_tile_p::MMAS_N <= 64);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[0], tidx);

    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // The base pointer of smem_v;
    char *smem_v_ = nullptr;
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        smem_v_ = &smem_[0];
    } else {
        smem_v_ = &smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE];
    }
    static_assert(Kernel_traits::SHARE_SMEM_FOR_K_AND_V);
    static_assert(Smem_tile_k::BYTES_PER_TILE == Smem_tile_v::BYTES_PER_TILE);
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[Smem_tile_v::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Smem_tile_v::BYTES_PER_TILE], tidx);

    // Trigger the loads for Q.
    gmem_q.load(smem_q);
    // Trigger the loads for K.
    gmem_k.load(smem_k);
    // Trigger the loads for K.
    gmem_v.load(smem_v);


    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_q);
    gmem_k.commit(smem_k);

    // Commit the data for V to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_v.commit(smem_v);
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // Load the fragments for Q.
    typename Smem_tile_q::Fragment frag_q[1][Mma_tile_p::MMAS_M];

    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_k::Fragment frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
        smem_k.load(frag_k[ki], ki);
    }

    // Commit the data for V to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_v.commit(smem_v);

    }

    enum { BITS_PER_ELT_S = sizeof(typename fmha::A_type) * 8 };

    Gmem_tile_s gmem_s(params.s_ptr, params, tidx);

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax< Cta_tile_p, Kernel_traits>;
    Softmax softmax(params, &smem_[Smem_tile_v::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE], bidb, tidx);

    constexpr int SMEM_BYTES_SOFTMAX = Softmax::ELEMENTS * sizeof(float);
    static_assert(SMEM_BYTES_SOFTMAX == Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float));

    enum { THREADS_PER_ROW = 32 };

    const float pinv = 1.f / params.p_dropout;

    // Load over the entire sequence length.
    for( int loop = 0, outer = 0; loop < Cta_tile_p::N; loop += Cta_tile_p::M, outer++ ) {
        if( loop >= binfo.actual_seqlen )
            break;

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {

            // Trigger the load from shared memory for the next series of Q values.
            smem_q.load(frag_q[0], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_p, frag_q[0], frag_k[ki]);
        }

        // Load the mask for that iteration.
        mask.load(outer);

        // Convert from the accumulator typ e to FP32 for Softmax.
        softmax.unpack(acc_p);

        // Apply the mask.
        softmax.apply_mask(mask);

        static_assert(2 * Mma_tile_p::MMAS_M * 4 * Mma_tile_p::MMAS_N <= 64);

        // Compute the max.
        float p_max[Mma_tile_p::MMAS_M * 2];
        softmax.template reduce<fmha::Max_>(p_max);

        // Make sure we are done reading shared memory.
        __syncthreads();
        // Compute the exponential value.
        softmax.apply_exp(p_max);
        // Compute the sum.
        float p_sum[Mma_tile_p::MMAS_M * 2];
        softmax.template reduce<fmha::Sum_>(p_sum);

        // Finalize softmax on the accumulators of P^T.
        softmax.scale(p_sum);

        __syncthreads();
        if( Is_training ) {
            auto encode_dropout = [](bool keep, float val) { return keep ? val : -val; };
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_p::MMAS_M; mi++ ) {
                #pragma unroll
                for( int ii = 0; ii < 2; ii++ ) {
                    #pragma unroll
                    for( int ni = 0; ni < Mma_tile_p::MMAS_N; ni++ ) {
                        float4 tmp = uniform4(ph());
                        // We encode the dropout pattern in the sign bit of the non-negative softmax to distinguish from
                        // pre-existing zeros
                        softmax.elt_[2 * mi + ii][4 * ni + 0] =
                            encode_dropout(tmp.x <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 0]);
                        softmax.elt_[2 * mi + ii][4 * ni + 1] =
                            encode_dropout(tmp.y <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 1]);
                        softmax.elt_[2 * mi + ii][4 * ni + 2] =
                            encode_dropout(tmp.z <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 2]);
                        softmax.elt_[2 * mi + ii][4 * ni + 3] =
                            encode_dropout(tmp.w <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 3]);
                    }
                }
            }

            gmem_s.store(softmax.elt_, mask);
            gmem_s.move();
        }

        // Trigger the load for the next Q values.
        if( loop + Cta_tile_p::M < Cta_tile_p::N ) {
            smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_q);
        }
        typename Smem_tile_v::Fragment frag_v[1][Mma_tile_o::MMAS_N];

        using Frag_p = fmha::Fragment_a< fmha::Row>;
        Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        softmax.pack(frag_p);
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ki++ ) {
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_o::MMAS_M; mi++ ) {
                #pragma unroll
                for( int ii = 0; ii < Frag_p::NUM_REGS; ii++ ) {
                    //"Apply" the dropout.
                    frag_p[ki][mi].reg(ii) = fmha::hmul2(frag_p[ki][mi].reg(ii), params.scale_dropout);
                    frag_p[ki][mi].reg(ii) = fmha::hrelu2(frag_p[ki][mi].reg(ii));
                }
            }
        }

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of V values.
            smem_v.load(frag_v[0], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_o, frag_p[ki], frag_v[0]);
        }

        // Loop over MMAS_M.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii ) {

            // Swizzle the elements and do the final reduction.
            smem_o.store(acc_o, ii);

            // Make sure the data is in shared memory.
            __syncthreads();

            // Load from shared memory.
            uint4 out[Gmem_tile_o::STGS_PER_LOOP];
            smem_o.load(out);

            // Always sync after last iter: shared smem_q and smem_o!
            __syncthreads();

            // Output the values.
            gmem_o.store(out, ii);
        }
        // same smem as o

        // Move to the next part of the output.
        gmem_o.move();

        // Commit the values for Q into shared memory.
        if( loop + Cta_tile_p::M < Cta_tile_p::N ) {
            gmem_q.commit(smem_q);
        }

        // Make sure the data is in shared memory.
        __syncthreads();

    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
