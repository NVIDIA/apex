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

#include <multihead_attn/philox.h>

#include <fmha.h>
#include <fmha/utils.h>
#include <fmha/smem_tile.h>
#include <fmha/gmem_tile.h>
#include <fmha/mask.h>
#include <fmha/softmax.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS_PER_CTA>
struct BlockInfoPadded {

    template<typename Params>
    __device__ BlockInfoPadded(const Params &params,
                               const int bidb,
                               const int bidh,
                               const int tidx)
        : bidb(bidb), bidh(bidh), h(params.h) {

        // The block index.
        sum_s = params.cu_seqlens[bidb];
        actual_seqlen = params.cu_seqlens[bidb + 1] - sum_s;
        bidx = sum_s * params.h + bidh;

        tidx_global = (bidb * params.h + bidh) * THREADS_PER_CTA + tidx;
    }

    __device__ bool stop_early() const {
        return actual_seqlen == 0;
    }

    int actual_seqlen;
    int bidx;
    int sum_s;
    int bidh;
    int bidb;
    int tidx_global;
    int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int CHUNKS, typename Cta_tile> 
struct Noloop_traits{
    // Interpretation of Cta_tile dims, i.e. Cta_tile_p:
    enum{ STEP = Cta_tile::M };
    enum{ SEQLEN = Cta_tile::N };

    // The size of the subsequence this CTA is processing
    enum { SUBSEQ = SEQLEN / CHUNKS };
    static_assert(SUBSEQ * CHUNKS == SEQLEN);

    // The number of steps to process the subsequence
    enum { NUM_STEPS = SUBSEQ / STEP };
    static_assert(NUM_STEPS  * Cta_tile::M == SUBSEQ);

    inline __device__ Noloop_traits(const int bidc) 
        : loop_offset_(NUM_STEPS * bidc)
        , bidc_(bidc) {
    }

    template<typename ... Tiles> 
    inline __device__ void move_all(Tiles & ... tiles) const {
        using expand_type = int[];
        for( int s = 0; s < loop_offset_; s++ ) {
            expand_type{ (tiles.move(), 0)... };
        }
    }

    inline __device__ int get_idx_dk() const {
        //return bidc_;
        return bidc_ * 2 + 0;
    }

    inline __device__ int get_idx_dv() const {
        //return CHUNKS + bidc_;
        return bidc_ * 2 + 1;
    }

    inline __device__ int offset_loop_count(const int l) {
        // convert loop counter to position in the outer sequence
        return (loop_offset_ + l) * STEP;
    }

    const int loop_offset_;
    const uint32_t bidc_;
    const int num_steps_ = NUM_STEPS;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile> 
struct Noloop_traits<3, Cta_tile>{
    // Interpretation of Cta_tile dims, i.e. Cta_tile_p:
    enum{ STEP = Cta_tile::M };
    enum{ SEQLEN = Cta_tile::N };

    static_assert(STEP == 16 && SEQLEN == 512);

    inline __device__ Noloop_traits(const int bidc)
        : bidc_(bidc)
        , num_steps_(bidc < 2 ? 11 : 10) 
        , loop_offset_(bidc * 11) {
    }

    template<typename ... Tiles> 
    inline __device__ void move_all(Tiles & ... tiles) const {
        using expand_type = int[];
        for( int s = 0; s < loop_offset_; s++ ) {
            expand_type{ (tiles.move(), 0)... };
        }
    }

    inline __device__ int get_idx_dk() const {
        //return bidc_;
        return bidc_ * 2 + 0;
    }

    inline __device__ int get_idx_dv() const {
        //return CHUNKS + bidc_;
        return bidc_ * 2 + 1;
    }

    inline __device__ int offset_loop_count(const int l) {
        // convert loop counter to position in the outer sequence
        return (loop_offset_ + l) * STEP;
    }

    const int loop_offset_;
    const uint32_t bidc_;
    const int  num_steps_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
