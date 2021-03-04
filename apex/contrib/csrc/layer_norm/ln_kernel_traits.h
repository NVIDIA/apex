#pragma once

constexpr uint32_t THREADS_PER_WARP = 32;

template <typename dtype, int COLS_, int WARPS_M_, int WARPS_N_,
          int BYTES_PER_LDG_ = 16>
struct Kernel_traits {
  enum { WARPS_M = WARPS_M_ };
  enum { WARPS_N = WARPS_N_ };
  enum { COLS = COLS_ };
  enum { BYTES_PER_LDG = BYTES_PER_LDG_ };

  using Vec = Vec<dtype, BYTES_PER_LDG>;

  using vec_t = typename Vec::vec_t;
  using base_t = typename Vec::base_t;
  using packed_t = typename Vec::packed_t;
  using compute_t = typename Vec::compute_t;
  using packed_compute_t = typename Vec::packed_compute_t;

  enum { THREADS_PER_ROW = WARPS_N * THREADS_PER_WARP };
  enum { THREADS_PER_CTA = WARPS_M * THREADS_PER_ROW };
  enum { ROWS_PER_CTA = WARPS_M };

  enum { BYTES_PER_ROW = COLS * sizeof(base_t) };
  enum { BYTES_PER_ROW_PER_CTA = THREADS_PER_ROW * BYTES_PER_LDG };
  enum {SMEM_BYTES = ROWS_PER_CTA * COLS * sizeof(compute_t)};
};
