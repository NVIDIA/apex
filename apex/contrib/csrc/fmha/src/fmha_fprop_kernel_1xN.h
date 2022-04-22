/***************************************************************************************************
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

template<typename Kernel_traits>
struct Gemm_Q_K_base {
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Fragment_q = typename Smem_tile_q::Fragment;
    using Fragment_k = typename Smem_tile_k::Fragment;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    static constexpr int SMEM_BYTES_SOFTMAX = Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float) * 2;

    __device__ inline Gemm_Q_K_base(char * smem_ptr_q, char * smem_ptr_k, const int tidx) 
        : smem_q(smem_ptr_q, tidx)
        , smem_k(smem_ptr_k, tidx) {

    }

    __device__ inline void load_q() {
        smem_q.load(frag_q[0], 0);
    }

    __device__ inline void reload_q() {
        smem_q.load(frag_q[0], 0);
    }

    Fragment_q frag_q[2][Mma_tile_p::MMAS_M];
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
};

template<typename Kernel_traits, bool K_in_regs>
struct Gemm_Q_K : public Gemm_Q_K_base<Kernel_traits> {

    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;

    enum { SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V };

    enum { SMEM_OFFSET_O = Smem_tile_q::BYTES_PER_TILE };
    enum { SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE) };

    // Q | K / V
    //   | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE 
                                    + std::max((SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE,
                                               Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX);

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            Base::smem_k.load(frag_k[ki], ki);
        }
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
    }

    __device__ inline void reload_k(){
        // Noop.
    }

    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
};


template<typename Kernel_traits>
struct Gemm_Q_K<Kernel_traits, false> : public Gemm_Q_K_base<Kernel_traits> {
    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;
    Fragment_k frag_k[2][Mma_tile_p::MMAS_N];

    enum { SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V };

    enum { SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE) };
    static_assert(Smem_tile_v::BYTES_PER_TILE == (int) Smem_tile_k::BYTES_PER_TILE);
    enum { SMEM_OFFSET_O = SMEM_OFFSET_V + Smem_tile_v::BYTES_PER_TILE };

    // Q | K/V + O + SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
                                    + (SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE 
                                    + Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX;

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
      : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        Base::smem_k.load(frag_k[0], 0);
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            Base::smem_k.load(frag_k[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
    }

    __device__ inline void reload_k(){
        Base::smem_k.load(frag_k[0], 0);
    }
};

template<typename Kernel_traits>
constexpr size_t get_dynamic_smem_size(){
    return Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>::SMEM_BYTES;
}

template<typename Kernel_traits, bool Is_training, typename Params, typename Prng>
inline __device__ void device_1xN_(const Params &params, const int bidb, const int bidh, const int begin, const int steps, Prng & ph) {


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

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;


    // The number of threads per row.
    enum { THREADS_PER_ROW = 32 };

    enum { BITS_PER_ELT_S = sizeof(fmha::A_type) * 8 };

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early() ) return;

    Gemm1 gemm_q_k(smem_, tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);
    // Wind gmem tiles to the correct position.
    for( int it = 0; it < begin; it++ ) {
        gmem_q.move();
        gmem_s.move();
        gmem_o.move();
    }

    fmha::Mask<Cta_tile_p> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // The base pointer of smem_v;
    char *smem_v_ = &smem_[Gemm1::SMEM_OFFSET_V];
    
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Gemm1::SMEM_OFFSET_O], tidx);

    // Trigger the loads for K.
    gmem_k.load(gemm_q_k.smem_k);
    // Trigger the loads for Q.
    gmem_q.load(gemm_q_k.smem_q);
    // Trigger the loads for V.
    gmem_v.load(smem_v);

    const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    #pragma unroll
    for(int it=0;it < Gmem_tile_k::LDGS;it++){
        gmem_k.fetch_[it] = fmha::hmul8(scale_bmm1, gmem_k.fetch_[it]);
    }



    // Commit the data for Q and V to shared memory.
    gmem_q.commit(gemm_q_k.smem_q);
    gmem_v.commit(smem_v);

    // Commit the data for K to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_k.commit(gemm_q_k.smem_k);
    }

    __syncthreads();

    // Load the fragments for Q.
    gemm_q_k.load_q();

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
        smem_v.load(frag_v[ki], ki);
    }

    // Commit the data for V to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_k.commit(gemm_q_k.smem_k);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for K. 
    gemm_q_k.load_k();

    // Create the object to do the softmax.
    Softmax softmax(params, &smem_[Gemm1::SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE], bidb, tidx);

    // Load over the entire sequence length.
    for( int l = 0; l < steps; l++ ) {
        if(begin + l * Cta_tile_p::M >= binfo.actual_seqlen) break;

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Do this part of P^T = (Q * K^T)^T.
        gemm_q_k(acc_p);

        // Trigger the load for the next Q values.
        if( l < steps - 1) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(gemm_q_k.smem_q);
        }

        // Load the mask for that iteration.
        mask.load(begin + l);

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p);

        // Apply the mask.
        softmax.apply_mask(mask);

        if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0 ) {
            // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
            __syncthreads();
        }
        // Compute the max.
        float p_max[Mma_tile_p::MMAS_M * 2];
        //softmax.template reduce<fmha::Max_>(p_max);
        softmax.reduce_max(p_max);

        // Compute the exponential value.
        softmax.apply_exp(p_max);

        // Compute the sum.
        float p_sum[Mma_tile_p::MMAS_M * 2];
        softmax.reduce_sum(p_sum);

        // Finalize softmax on the accumulators of P^T.
        softmax.scale(p_sum);

        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        if( Is_training ) {
            auto encode_dropout = [](bool keep, float val) { return keep ? val : -val; };
            #pragma unroll
            for( int mi = 0; mi < Mma_tile_p::MMAS_M; mi++ ) {
                #pragma unroll
                for( int ii = 0; ii < 2; ii++ ) {
                    #pragma unroll
                    for( int ni = 0; ni < Mma_tile_p::MMAS_N; ni++ ) {
                        float4 tmp = uniform4(ph());
                        // We encode the dropout pattern in the sign bit of the non-negative softmax to distinguish from pre-existing zeros
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
            softmax.pack(frag_p);
            gmem_s.store(frag_p, mask);
            gmem_s.move();
        } else {
            softmax.pack(frag_p);
        }

        // Commit the values for Q into shared memory.
        if(l < steps - 1) {
            gmem_q.commit(gemm_q_k.smem_q);
        }

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

        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
            fmha::gemm(acc_o, frag_p[ki], frag_v[ki]);
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

            // Make sure the data was read from shared memory.
            if( ii < Gmem_tile_o::LOOPS - 1 ) {
                __syncthreads();
            }

            // Output the values.
            gmem_o.store(out, ii);
        }

        // Move to the next part of the output.
        gmem_o.move();
        gemm_q_k.reload_k();

        // Commit the values for Q into shared memory.
        if(l < steps - 1) {
            gemm_q_k.reload_q();
        }

    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_training, typename Params>
inline __device__ void device_1xN(const Params &params, 
                                  const int num_full_heads,
                                  const int num_main_groups,
                                  const int main_group_size,
                                  const int main_steps,
                                  const int rest_steps) {

    constexpr int STEPS = Kernel_traits::Cta_tile_p::N / Kernel_traits::Cta_tile_p::M;
    const int tidx_global = blockIdx.x * gridDim.x + threadIdx.x;
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph(std::get<0>(seeds), tidx_global, std::get<1>(seeds));
    for( int it = 0; it < num_full_heads; it++ ) {
        const int bidx = it * gridDim.x + blockIdx.x;
        const int bidh = bidx % params.h;
        const int bidb = bidx / params.h;
        fmha::device_1xN_<Kernel_traits, Is_training>(params, bidb, bidh, 0, STEPS, ph);
        __syncthreads();
    }
    if( main_group_size == 0 )
        return;
    const int head_offset = num_full_heads * gridDim.x;

    if( blockIdx.x < main_group_size * num_main_groups ) {
        // process within heads
        const int group = blockIdx.x % num_main_groups;
        const int bidx = blockIdx.x / num_main_groups;
        const int bidh = (head_offset + bidx) % params.h;
        const int bidb = (head_offset + bidx) / params.h;
        const int offset = group * main_steps;
        fmha::device_1xN_<Kernel_traits, Is_training>(params, bidb, bidh, offset, main_steps, ph);
    } else {
        if(rest_steps == 0 ) return;
        // process across heads
        const int bidx = blockIdx.x - main_group_size * num_main_groups;
        const int offset = num_main_groups * main_steps;
        const int total_heads = params.b * params.h;
        const int rest_ctas = gridDim.x - main_group_size * num_main_groups;
        for( int it = head_offset + bidx; it < total_heads; it += rest_ctas ) {
            const int bidh = it % params.h;
            const int bidb = it / params.h;
            fmha::device_1xN_<Kernel_traits, Is_training>(params, bidb, bidh, offset, rest_steps, ph);
            __syncthreads();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_training, typename Params>
inline __device__ void device_1xN(const Params &params, const int total_heads) {

    const int tidx_global = blockIdx.x * gridDim.x + threadIdx.x;
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph(std::get<0>(seeds), tidx_global, std::get<1>(seeds));
    constexpr int STEPS = Kernel_traits::Cta_tile_p::N / Kernel_traits::Cta_tile_p::M;

    for(int bidx = blockIdx.x; bidx < total_heads; bidx += gridDim.x){
        const int bidh = bidx % params.h;
        const int bidb = bidx / params.h;
        fmha::device_1xN_<Kernel_traits, Is_training>(params, bidb, bidh, 0, STEPS, ph);
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

