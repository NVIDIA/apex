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

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Sum_ {
    enum { IS_SUM = 1 };
    static inline __device__ float apply(float x, float y) {
        return x + y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Max_ {
    enum { IS_SUM = 0 };
    static inline __device__ float apply(float x, float y) {
        return x > y ? x : y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float apply_exp_(float x, float max) {
    return __expf(x - max);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int COLS> struct ReadType {};
template<> struct ReadType<4> { using T = float;};
template<> struct ReadType<8> { using T = float2;};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Smem_tile_reduce {
    // Helper class to distribute MMA tiles reduced over rows per warp over quads.

    // The Mma tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum { MMAS_M = Mma_tile::MMAS_M };
    enum { MMAS_N = Mma_tile::MMAS_N };

    enum { WARPS_M = Cta_tile::WARPS_M };
    enum { WARPS_N = Cta_tile::WARPS_N };


    static constexpr int ROWS = WARPS_M * MMAS_M * 16;
    static constexpr int COLS = WARPS_N;
    static_assert(COLS == 4 || COLS == 8);
    static constexpr int ROWS_PER_XOR_PATTERN = (COLS == 8) ? 4 : 8;
    static constexpr int BYTES_PER_TILE = ROWS * COLS * sizeof(float);
    static constexpr int ELTS_PER_TILE = ROWS * COLS;

    static constexpr int THREADS_PER_GROUP = Kernel_traits::Gmem_tile_o::THREADS_PER_ROW;
    static_assert(THREADS_PER_GROUP == 16); // DEBUG
    static constexpr int ROWS_PER_WARP = 32 / THREADS_PER_GROUP;
    static constexpr int LOOPS = Kernel_traits::Gmem_tile_o::LOOPS;
    static_assert(LOOPS == 1);

    using read_t = typename ReadType<COLS>::T;

    __device__ inline Smem_tile_reduce(float *smem_, const int tidx) {

        int lane = tidx % 32;
        int warp = tidx / 32;

        int warp_m = warp % WARPS_M;
        int warp_n = warp / WARPS_M;

        qid_ = lane % 4;
        int qp = lane / 4;

        // Swizzle the column to avoid 2-fold bank conflicts when we have 8 warps.
        // This won't affect reading as we assume commutative reduction ops.
        const int col = warp_n ^ (qp / ROWS_PER_XOR_PATTERN);
        smem_write_ = &smem_[warp_m * 16 * MMAS_M * WARPS_N + qp * WARPS_N + col];
        smem_read_ = &reinterpret_cast<read_t *>(smem_)[warp_m * 16 * MMAS_M * 4 + qp * 4 + qid_];

    }

    __device__ inline void store(float (&frag)[2 * MMAS_M]) {
        if( qid_ == 0 ) {
            #pragma unroll
            for( int mi = 0; mi < MMAS_M; mi++ ) {
                int offset = mi * 16 * WARPS_N;
                smem_write_[offset + 0 * 8 * WARPS_N] = frag[mi * 2 + 0];
                smem_write_[offset + 1 * 8 * WARPS_N] = frag[mi * 2 + 1];
            }
        }
    }

    __device__ inline void load(read_t (&frag)[2 * MMAS_M]) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; mi++ ) {
            int offset = mi * 16 * 4;
            frag[mi * 2 + 0] = smem_read_[offset + 0 * 8 * 4];
            frag[mi * 2 + 1] = smem_read_[offset + 1 * 8 * 4];
        }
    }

    int qid_;
    float *smem_write_;
    read_t *smem_read_;

};


template<typename Cta_tile, typename Kernel_traits>
struct Softmax_base {

    // The Mma tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum { MMAS_M = Mma_tile::MMAS_M };
    enum { MMAS_N = Mma_tile::MMAS_N };

    // The number of groups of warp such that we have at most 4 warps writing consecutive elements.
    enum { GROUPS = fmha::Div_up<Cta_tile::WARPS_N, 4>::VALUE };
    // The number of elements that we are going to store per row.
    enum { ELEMENTS_PER_ROW = Cta_tile::WARPS_N / GROUPS };
    // The number of rows.
    enum { ROWS = Cta_tile::M * GROUPS };
    // The total number of elements.
    enum { ELEMENTS = ROWS * ELEMENTS_PER_ROW };

    // Ctor.
    template<typename Params>
    inline __device__ Softmax_base(const Params &params, void *smem, int bidb, int tidx)
        :  // packed_mask_ptr_(reinterpret_cast<const char*>(params.packed_mask_ptr)),
          smem_(reinterpret_cast<float *>(smem)), tidx_(tidx) {

        // Move to the 1st mask loaded by the thread+ tidx;
        // packed_mask_ptr_ += bidb * params.packed_mask_stride_in_bytes + tidx * sizeof(uint32_t);

        // Extract the position in the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // Decompose the warp index into M and N.
        int warp_m = warp % Cta_tile::WARPS_M;
        int warp_n = warp / Cta_tile::WARPS_M;

        // Decompose the warp-n index into group/position-inside-the-group.
        int warp_g = warp_n / ELEMENTS_PER_ROW;
        int warp_i = warp_n % ELEMENTS_PER_ROW;

        // The location written by the threads.
        int write_row = warp_g * (ROWS / GROUPS) + warp_m * Mma_tile::M_PER_MMA + lane / 4;
        int write_col = warp_i;

        // Assemble the write pointer.
        smem_write_ = &smem_[write_row * ELEMENTS_PER_ROW + write_col];

        // Assemble the read pointer.
        smem_read_ = &smem_[warp_m * Mma_tile::M_PER_MMA + lane / 4];
    }

    template<typename Mask>
    inline __device__ void apply_mask(const Mask &mask) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; ++mi ) {
            #pragma unroll
            for( int ii = 0; ii < 2; ++ii ) {
                #pragma unroll
                for( int ni = 0; ni < MMAS_N; ++ni ) {
                    #pragma unroll
                    for( int jj = 0; jj < 4; ++jj ) {
                        if( !mask.is_valid(mi, ni, ii, jj) ) {
                            elt_[2 * mi + ii][4 * ni + jj] = -INFINITY;
                        }
                    }
                }
            }
        }
    }

    // Apply the exp to all the elements.
    inline __device__ void apply_exp(const float (&max)[MMAS_M * 2]) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M * 2; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < MMAS_N * 4; ++ni ) {
                elt_[mi][ni] = apply_exp_(elt_[mi][ni], max[mi]);
            }
        }
    }

    // Scale all the elements.
    inline __device__ void scale(const float (&sum)[MMAS_M * 2]) {
        // Precompute the inverse sum to normalize. Without -use_fast_math, it makes a huge deal.
        float inv_sum[MMAS_M * 2];
        #pragma unroll
        for( int mi = 0; mi < MMAS_M * 2; ++mi ) {
            inv_sum[mi] = (sum[mi] == 0.f || sum[mi] != sum[mi]) ? 1.f : 1.f / sum[mi];
        }

        // Update the values.
        #pragma unroll
        for( int mi = 0; mi < MMAS_M * 2; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < MMAS_N * 4; ++ni ) {
                elt_[mi][ni] *= inv_sum[mi];
            }
        }
    }

    // The pointer to the mask.
    const char *packed_mask_ptr_;
    // Shared memory for the CTA-wide reduction.
    float *smem_, *smem_write_, *smem_read_;
    // The current thread index.
    int tidx_;
    // The elements.
    float elt_[MMAS_M * 2][MMAS_N * 4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile, typename Kernel_traits>
struct Softmax : public Softmax_base<Cta_tile, Kernel_traits> {

    // The base class.
    using Base = Softmax_base<Cta_tile, Kernel_traits>;
    // The fragment.
    using Fragment_a = fmha::Fragment_a<fmha::Row>;

    static_assert(Fragment_a::NUM_REGS == 4);

    enum { WARPS_M = Cta_tile::WARPS_M };
    enum { WARPS_N = Cta_tile::WARPS_N };
    // The MMAs.
    enum { MMAS_M = Base::MMAS_M };
    enum { MMAS_N = Base::MMAS_N };

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator;
    using Accumulator_out = Fragment<uint16_t, 8>;
    static_assert(Accumulator_out::NUM_REGS == 4);

    static_assert(std::is_same<Accumulator::Data_type, float>::value);

    using Smem_tile_red = Smem_tile_reduce<Cta_tile, Kernel_traits>;
    static_assert(Smem_tile_red::ELTS_PER_TILE == Cta_tile::M * WARPS_N);
    // Ctor.
    template<typename Params>
    inline __device__ Softmax(const Params &params, void *smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_bmm1_(params.scale_bmm1) 
        , smem_sum_(static_cast<float*>(smem), tidx)
        , smem_max_(static_cast<float*>(smem) + Smem_tile_red::ELTS_PER_TILE, tidx) {
    }

    // Pack the data to a fragment for the next GEMM.
    template<int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const {
        #pragma unroll
        for( int mi = 0; mi < M; ++mi ) {
            #pragma unroll
            for( int ki = 0; ki < K; ++ki ) {

                // 1st row - 4 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0];
                float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1];
                float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2];
                float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3];

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0];
                float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1];
                float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2];
                float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3];

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float2_to_half2(tmp_00, tmp_01);
                dst[ki][mi].reg(1) = fmha::float2_to_half2(tmp_10, tmp_11);
                dst[ki][mi].reg(2) = fmha::float2_to_half2(tmp_02, tmp_03);
                dst[ki][mi].reg(3) = fmha::float2_to_half2(tmp_12, tmp_13);
            }
        }
    }

    // Scale FP32 fragments
    inline __device__ void unpack(const Accumulator (&acc)[MMAS_M][MMAS_N]) {
        const float scalef = reinterpret_cast<const float &>(this->params_scale_bmm1_);

        #pragma unroll
        for( int mi = 0; mi < MMAS_M; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < MMAS_N; ++ni ) {
                // 1st row - 4 elements per row.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0) * scalef;
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1) * scalef;
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4) * scalef;
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5) * scalef;
                // 2nd row - 4 elements per row.
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2) * scalef;
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3) * scalef;
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6) * scalef;
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7) * scalef;
            }
        }
    }
    // Scale FP32 fragments
    inline __device__ void unpack_noscale(const Accumulator (&acc)[MMAS_M][MMAS_N]) {

        #pragma unroll
        for( int mi = 0; mi < MMAS_M; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < MMAS_N; ++ni ) {
                // 1st row - 4 elements per row.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0);
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1);
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4);
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5);
                // 2nd row - 4 elements per row.
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2);
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3);
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6);
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7);
            }
        }
    }



    template<typename Operator>
    __device__ inline void reduce_(float (&frag)[2 * MMAS_M], Operator &op, Smem_tile_red & smem_red) {
        for( int mi = 0; mi < 2 * MMAS_M; mi++ ) {
            frag[mi] = this->elt_[mi][0];
            for( int ni = 1; ni < 4 * MMAS_N; ni++ ) {
                frag[mi] = op(frag[mi], this->elt_[mi][ni]);
            }
        }
        quad_reduce(frag, frag, op);

        smem_red.store(frag);
        __syncthreads();
        typename Smem_tile_red::read_t tmp[2 * MMAS_M];
        smem_red.load(tmp);

        quad_allreduce(frag, tmp, op);
    }

    __device__ inline void reduce_max(float (&frag)[2 * MMAS_M]){ 
        MaxOp<float> max;
        reduce_(frag, max, smem_max_);
    }

    __device__ inline void reduce_sum(float (&frag)[2 * MMAS_M]){ 
        SumOp<float> sum;
        reduce_(frag, sum, smem_sum_);
    }


    const uint32_t params_scale_bmm1_;
    Smem_tile_red smem_max_;
    Smem_tile_red smem_sum_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
