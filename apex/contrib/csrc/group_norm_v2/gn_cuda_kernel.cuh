#pragma once

#include <cooperative_groups.h>

#include "gn_utils.hpp"

namespace group_norm_v2 {

namespace cg = cooperative_groups;

template <typename T>
inline constexpr T up_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
inline constexpr T round_up(T a, T b) {
  return up_div(a, b) * b;
}

inline constexpr unsigned round_up_pow2(unsigned x) {
  int log = 0;
  x--;
  while (x) {
    x /= 2;
    log++;
  }
  return 1U << log;
}

inline constexpr unsigned round_down_pow2(unsigned x) { return round_up_pow2(x + 1) / 2; }

template <typename T>
inline constexpr T gcd(T a, T b) {
  while (b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

template <typename T>
inline constexpr T lcm(T a, T b) {
  return (a * b) / gcd(a, b);
}

template <typename T>
inline constexpr T relative_prime(T x, T min) {
  int p = min;
  while (gcd(p, x) != 1) {
    p++;
  }
  return p;
}

template <typename T>
inline constexpr T max_divisor(T x, T max) {
  int p = max;
  while (x % p != 0) {
    p--;
  }
  return p;
}

constexpr unsigned FINAL_MASK = 0xffffffff;

template <int VIRTUAL_CLUSTER_SIZE, bool PERSISTENT, bool HARDWARE_CLUSTER>
__device__ void virtual_cluster_sync(unsigned int* barrier) {
  if constexpr (VIRTUAL_CLUSTER_SIZE == 1) {
    __syncthreads();
  } else if constexpr (HARDWARE_CLUSTER) {
    cg::this_cluster().sync();
  } else {
    static_assert(PERSISTENT, "potential deadlock");
    volatile unsigned int* arrived = &barrier[blockIdx.y];
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int expected = VIRTUAL_CLUSTER_SIZE;
      bool gpu_master = blockIdx.x == 0;
      unsigned int nb = 1;
      if (gpu_master) {
        nb = 0x80000000 - (expected - 1);
      }
      unsigned int oldArrive;
      asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;"
                   : "=r"(oldArrive)
                   : _CG_ASM_PTR_CONSTRAINT((unsigned int*)arrived), "r"(nb)
                   : "memory");
      unsigned int current_arrive;
      do {
        asm volatile("ld.acquire.gpu.u32 %0,[%1];"
                     : "=r"(current_arrive)
                     : _CG_ASM_PTR_CONSTRAINT((unsigned int*)arrived)
                     : "memory");
      } while (!cooperative_groups::details::bar_has_flipped(oldArrive, current_arrive));
    }
    __syncthreads();
  }
}

template <int NUM_BLOCKS, bool PERSISTENT>
__device__ unsigned int group_barrier_arrive(unsigned int* barrier, bool gpu_master) {
  static_assert(PERSISTENT, "potential deadlock");
  volatile unsigned int* arrived = &barrier[0];
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int expected = NUM_BLOCKS;
    unsigned int nb = 1;
    if (gpu_master) {
      nb = 0x80000000 - (expected - 1);
    }
    unsigned int oldArrive;
    asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;"
                 : "=r"(oldArrive)
                 : _CG_ASM_PTR_CONSTRAINT((unsigned int*)arrived), "r"(nb)
                 : "memory");
    return oldArrive;
  } else {
    return 0;
  }
}

__device__ inline void group_barrier_wait(unsigned int* barrier, unsigned int oldArrive) {
  volatile unsigned int* arrived = &barrier[0];
  if (threadIdx.x == 0) {
    unsigned int current_arrive;
    do {
      asm volatile("ld.acquire.gpu.u32 %0,[%1];"
                   : "=r"(current_arrive)
                   : _CG_ASM_PTR_CONSTRAINT((unsigned int*)arrived)
                   : "memory");
    } while (!cooperative_groups::details::bar_has_flipped(oldArrive, current_arrive));
  }
  __syncthreads();
}

// Calculate `n` (batch id) and `c` (channel range id) for each loop
template <bool CONSTANT_C_LOOP, int C, int C_PER_CLUSTER, int NUM_VIRTUAL_CLUSTERS, bool PERSISTENT>
class NCScheduler;

template <int C, int C_PER_CLUSTER, int NUM_VIRTUAL_CLUSTERS, bool PERSISTENT>
class NCScheduler<false, C, C_PER_CLUSTER, NUM_VIRTUAL_CLUSTERS, PERSISTENT> {
 public:
  __device__ NCScheduler(int64_t n) {
    nc_loop_ = blockIdx.y;
    at_end_ = nc_loop_ >= n * (C / C_PER_CLUSTER);
  }
  __device__ auto get_nc() {
    int64_t n_loop = nc_loop_ / (C / C_PER_CLUSTER);
    int c_loop = nc_loop_ % (C / C_PER_CLUSTER);
    return std::make_tuple(n_loop, c_loop);
  }
  __device__ void next(int64_t n) {
    if constexpr (PERSISTENT) {
      nc_loop_ += NUM_VIRTUAL_CLUSTERS;
      at_end_ = nc_loop_ >= n * (C / C_PER_CLUSTER);
    }
  }
  __device__ bool at_end(int64_t n) { return !PERSISTENT || at_end_; }

 private:
  int64_t nc_loop_;
  bool at_end_;
};

template <int C, int C_PER_CLUSTER, int NUM_VIRTUAL_CLUSTERS, bool PERSISTENT>
class NCScheduler<true, C, C_PER_CLUSTER, NUM_VIRTUAL_CLUSTERS, PERSISTENT> {
 public:
  __device__ NCScheduler(int64_t n) {
    n_loop_ = blockIdx.y / (C / C_PER_CLUSTER);
    c_loop_ = blockIdx.y % (C / C_PER_CLUSTER);
  }
  __device__ auto get_nc() { return std::make_tuple(n_loop_, c_loop_); }
  __device__ void next(int64_t n) {
    if constexpr (PERSISTENT) {
      n_loop_ += NUM_VIRTUAL_CLUSTERS / (C / C_PER_CLUSTER);
    }
  }
  __device__ bool at_end(int64_t n) { return !PERSISTENT || n_loop_ >= n; }

 private:
  int64_t n_loop_;
  int c_loop_;
};

class CompileConditionAlwaysTrue {
 public:
  __device__ static constexpr bool matches() { return true; }
};

template <typename T, int BLOCK_DIM_X, int BLOCKS_PER_SM, int G, int CPG, int HW, bool SILU, int ROWS_PER_BLOCK,
          int C_PER_BLOCK, int C_PER_CLUSTER, int VEC_ELEMS, bool PERSISTENT, int NUM_VIRTUAL_CLUSTERS, bool LOAD_TWICE,
          bool HARDWARE_CLUSTER, class CompileCondition = CompileConditionAlwaysTrue>
__global__ __launch_bounds__(BLOCK_DIM_X, BLOCKS_PER_SM) void gn_cuda_kernel(
    T* __restrict__ out, T const* __restrict__ x, T const* __restrict__ w, T const* __restrict__ b, float eps,
    int64_t n, float* __restrict__ mean_var_out, float* __restrict__ red_buffer, unsigned* __restrict__ barrier) {
  // Procedure Overview
  //   1. Thread sum: read from gmem, write partial sum to smem, store input in registers (if no LOAD_TWICE)
  //   2. Block sum: read from smem, write partial sum to gmem (or distributed shared memory if HARDWARE_CLUSTER is
  //   used)
  //   3. Group sum: read from gmem, write mean&var to smem
  //   4. Scale: read mean&var from smem, read input from gmem (if LOAD_TWICE), write output to gmem

  static_assert(BLOCK_DIM_X % 32 == 0, "warp shuffle error");

  constexpr int C = G * CPG;
  static_assert(C % C_PER_CLUSTER == 0, "cannot divide channels into clusters");
  static_assert(C_PER_CLUSTER % C_PER_BLOCK == 0, "cannot divide a cluster into blocks");
  static_assert(C_PER_CLUSTER % CPG == 0, "no reduce between clusters, would produce incorrect results");
  static_assert(!(C_PER_BLOCK % CPG == 0 && C_PER_CLUSTER != C_PER_BLOCK),
                "inefficient configuration, please reduce C_PER_CLUSTER");

  static_assert(ROWS_PER_BLOCK * C_PER_BLOCK % BLOCK_DIM_X == 0, "cannot divide tile into threads");
  struct alignas(VEC_ELEMS * sizeof(T)) U {
    T data[VEC_ELEMS];
  };

  auto compute_mean_var = [&](float2 sum) {
    float mean = sum.x / (HW * CPG);
    float var = std::max(0.f, sum.y / (HW * CPG) - mean * mean);
    return float2{mean, var};
  };

  static_assert(HW % ROWS_PER_BLOCK == 0,
                "HW must be divisible by ROWS_PER_BLOCK to determine the number of blocks on the HW axis");
  constexpr int MAX_NUM_GROUPS_PER_BLOCK =
      C_PER_BLOCK % CPG == 0 ? C_PER_BLOCK / CPG : up_div(C_PER_BLOCK - gcd(C_PER_BLOCK, CPG), CPG) + 1;
  constexpr int VIRTUAL_CLUSTER_SIZE = (C_PER_CLUSTER / C_PER_BLOCK) * (HW / ROWS_PER_BLOCK);
  constexpr int virtual_cluster_dim_x = C_PER_CLUSTER / C_PER_BLOCK;
  constexpr int virtual_cluster_dim_y = HW / ROWS_PER_BLOCK;
  int virtual_block_idx_x = (blockIdx.x % VIRTUAL_CLUSTER_SIZE) % virtual_cluster_dim_x;
  int virtual_block_idx_y = (blockIdx.x % VIRTUAL_CLUSTER_SIZE) / virtual_cluster_dim_x;

  if constexpr (CompileCondition::matches()) {
    int step = 0;
    constexpr bool CONSTANT_C_LOOP = PERSISTENT && NUM_VIRTUAL_CLUSTERS % (C / C_PER_CLUSTER) == 0;
    NCScheduler<CONSTANT_C_LOOP, C, C_PER_CLUSTER, NUM_VIRTUAL_CLUSTERS, PERSISTENT> nc_scheduler(n);
    while (true) {  // TODO: unroll the loop
      if constexpr (PERSISTENT) {
        if (nc_scheduler.at_end(n)) {
          break;
        }
      }
      auto [n_loop, c_loop] = nc_scheduler.get_nc();
      if constexpr (PERSISTENT) {
        nc_scheduler.next(n);
      }
      static_assert(C_PER_BLOCK % VEC_ELEMS == 0, "cannot vectorize");
      static_assert((BLOCK_DIM_X * VEC_ELEMS) % C_PER_BLOCK == 0,
                    "each block should load one or more C_PER_BLOCK at once");
      constexpr int ROWS_PER_IO = BLOCK_DIM_X * VEC_ELEMS / C_PER_BLOCK;
      static_assert(ROWS_PER_BLOCK % ROWS_PER_IO == 0, "cannot determine the IO times per batch");
      int block_channel_start = virtual_block_idx_x * C_PER_BLOCK + c_loop * C_PER_CLUSTER;
      int block_group_start = block_channel_start / CPG;
      int thread_channel_start = block_channel_start + threadIdx.x % (C_PER_BLOCK / VEC_ELEMS) * VEC_ELEMS;
      U frag[ROWS_PER_BLOCK / ROWS_PER_IO];

      // GCD_VEC_CPG is an important constant that determines how many channels can be merged in reduction computation
      //   For example, VEC_ELEMS=4 and CPG=10, then GCD_VEC_CPG=2,
      //   so we need to store only 2 sums on each thread, and compute only 2 mean&var for each thread.
      constexpr int GCD_VEC_CPG = gcd(VEC_ELEMS, CPG);

      // If each block handles only one group, run warpReduce and store the sum to `sum_per_channel_single_group`;
      // otherwise store (VEC_ELEMS / GCD_VEC_CPG) sums to `sum_per_channel_multi_group`, where `relative_prime` is used
      // for swizzle.
      constexpr bool SINGLE_GROUP_PER_BLOCK = CPG % C_PER_BLOCK == 0;
      [[maybe_unused]] __shared__ float2 sum_per_channel_single_group[BLOCK_DIM_X / 32];
      [[maybe_unused]] __shared__ float2 sum_per_channel_multi_group[C_PER_BLOCK / GCD_VEC_CPG][relative_prime(
          128 / (int)sizeof(float2), ROWS_PER_IO)];

      if constexpr (LOAD_TWICE) {
        float2 frag_sum_per_channel[VEC_ELEMS / GCD_VEC_CPG]{};
        for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
          int64_t input_idx =
              n_loop * HW * C +
              (virtual_block_idx_y * ROWS_PER_BLOCK + j * ROWS_PER_IO + threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)) * C +
              thread_channel_start;
          U val = *reinterpret_cast<U const*>(&x[input_idx]);
          for (int i = 0; i < VEC_ELEMS / GCD_VEC_CPG; i++) {
            float2 sum = frag_sum_per_channel[i];
            for (int k = 0; k < GCD_VEC_CPG; k++) {
              sum.x += (float)val.data[i * GCD_VEC_CPG + k];
              sum.y += (float)val.data[i * GCD_VEC_CPG + k] * (float)val.data[i * GCD_VEC_CPG + k];
            }
            frag_sum_per_channel[i] = sum;
          }
        }
        for (int i = 0; i < VEC_ELEMS / GCD_VEC_CPG; i++) {
          if constexpr (SINGLE_GROUP_PER_BLOCK) {
            for (int mask = 16; mask > 0; mask >>= 1) {
              frag_sum_per_channel[i].x += __shfl_xor_sync(FINAL_MASK, frag_sum_per_channel[i].x, mask, 32);
              frag_sum_per_channel[i].y += __shfl_xor_sync(FINAL_MASK, frag_sum_per_channel[i].y, mask, 32);
            }
            static_assert(VEC_ELEMS / GCD_VEC_CPG == 1, "process only one element for each warp");
            if (threadIdx.x % 32 == 0) {
              sum_per_channel_single_group[threadIdx.x / 32] = frag_sum_per_channel[i];
            }
          } else {
            sum_per_channel_multi_group[i * (C_PER_BLOCK / VEC_ELEMS) + threadIdx.x % (C_PER_BLOCK / VEC_ELEMS)]
                                       [threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)] = frag_sum_per_channel[i];
          }
        }
        __syncthreads();
      } else {
        for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
          int64_t input_idx =
              n_loop * HW * C +
              (virtual_block_idx_y * ROWS_PER_BLOCK + j * ROWS_PER_IO + threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)) * C +
              thread_channel_start;
          frag[j] = *reinterpret_cast<U const*>(&x[input_idx]);
        }

        for (int i = 0; i < VEC_ELEMS / GCD_VEC_CPG; i++) {
          float2 sum = {0.f, 0.f};
          for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
            for (int k = 0; k < GCD_VEC_CPG; k++) {
              sum.x += (float)frag[j].data[i * GCD_VEC_CPG + k];
              sum.y += (float)frag[j].data[i * GCD_VEC_CPG + k] * (float)frag[j].data[i * GCD_VEC_CPG + k];
            }
          }
          if constexpr (SINGLE_GROUP_PER_BLOCK) {
            for (int mask = 16; mask > 0; mask >>= 1) {
              sum.x += __shfl_xor_sync(FINAL_MASK, sum.x, mask, 32);
              sum.y += __shfl_xor_sync(FINAL_MASK, sum.y, mask, 32);
            }
            static_assert(VEC_ELEMS / GCD_VEC_CPG == 1, "process only one element for each warp");
            if (threadIdx.x % 32 == 0) {
              sum_per_channel_single_group[threadIdx.x / 32] = sum;
            }
          } else {
            sum_per_channel_multi_group[i * (C_PER_BLOCK / VEC_ELEMS) + threadIdx.x % (C_PER_BLOCK / VEC_ELEMS)]
                                       [threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)] = sum;
          }
        }
        __syncthreads();
      }

      U uw = *reinterpret_cast<U const*>(&w[thread_channel_start]);
      U ub = *reinterpret_cast<U const*>(&b[thread_channel_start]);

      // Three cases for the red_buffer:
      //   - Block sync (VIRTUAL_CLUSTER_SIZE=1): use shared memory
      //   - Virtual cluster sync with HARDWARE_CLUSTER: use distributed shared memory
      //   - Virtual cluster sync without HARDWARE_CLUSTER: use global memory, i.e., `red_buffer`
      constexpr bool USE_SHARED_RED_BUFFER = HARDWARE_CLUSTER || VIRTUAL_CLUSTER_SIZE == 1;

      // Specialize for the case that each group is handled by only one block
      //   For common cases, blockSum produces partial sum and stores it to the red_buffer, and groupSum produces
      //   mean&var For the special case, blockSum produces mean&var directly
      constexpr bool STORE_MEAN_VAR_IN_SHARED_RED_BUFFER =
          VIRTUAL_CLUSTER_SIZE == 1 &&
          MAX_NUM_GROUPS_PER_BLOCK == 1;  // MAX_NUM_GROUPS_PER_BLOCK > 1 is possible but not implemented

      [[maybe_unused]] __align__(16)
          __shared__ float2 shared_red_buffer[MAX_NUM_GROUPS_PER_BLOCK * (STORE_MEAN_VAR_IN_SHARED_RED_BUFFER ? 1 : 2)];

      // Block sum
      if constexpr (SINGLE_GROUP_PER_BLOCK) {
        // block reduce
        if (threadIdx.x < 32) {
          float2 sum_local_group =
              threadIdx.x < BLOCK_DIM_X / 32 ? sum_per_channel_single_group[threadIdx.x] : float2{0.f, 0.f};
          constexpr int warp_num_pow2 = round_up_pow2(BLOCK_DIM_X / 32);
          for (int mask = warp_num_pow2 / 2; mask > 0; mask >>= 1) {
            sum_local_group.x += __shfl_xor_sync(FINAL_MASK, sum_local_group.x, mask, 32);
            sum_local_group.y += __shfl_xor_sync(FINAL_MASK, sum_local_group.y, mask, 32);
          }
          if (threadIdx.x == 0) {
            if constexpr (USE_SHARED_RED_BUFFER) {
              if constexpr (STORE_MEAN_VAR_IN_SHARED_RED_BUFFER) {
                shared_red_buffer[0] = compute_mean_var(sum_local_group);
              } else {
                shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + 0] = sum_local_group;
              }
            } else {
              *reinterpret_cast<float2*>(
                  &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                               virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                               // (threadIdx.x / THREADS_PER_GROUP) * virtual_cluster_dim_y +
                               virtual_block_idx_y) *
                              2]) = sum_local_group;
            }
          }
        }
      } else {
        // The number of threads to calculate the sum of each group (should be a power of 2 for warp reduce)
        constexpr int THREADS_PER_GROUP = std::min(std::min(32U, round_up_pow2(ROWS_PER_IO)),
                                                   round_up_pow2(BLOCK_DIM_X / MAX_NUM_GROUPS_PER_BLOCK / 2 + 1));
        static_assert(BLOCK_DIM_X >= MAX_NUM_GROUPS_PER_BLOCK * THREADS_PER_GROUP, "not enough threads");
        float2 sum_local_group = {0.f, 0.f};
        if (threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          int local_group_idx = block_group_start + threadIdx.x / THREADS_PER_GROUP;
          // TODO: map threads to both the CPG loop and the ROWS loop
          for (int local_c_loop = 0; local_c_loop < CPG; local_c_loop += GCD_VEC_CPG) {
            int c = local_group_idx * CPG + local_c_loop;
            if (C_PER_BLOCK % CPG == 0 || (c >= block_channel_start && c < block_channel_start + C_PER_BLOCK)) {
              for (int src_thread_tile_y = threadIdx.x % THREADS_PER_GROUP; src_thread_tile_y < ROWS_PER_IO;
                   src_thread_tile_y += THREADS_PER_GROUP) {
                int channel_idx = (c - block_channel_start) / GCD_VEC_CPG;
                channel_idx = channel_idx % (VEC_ELEMS / GCD_VEC_CPG) * (C_PER_BLOCK / VEC_ELEMS) +
                              channel_idx / (VEC_ELEMS / GCD_VEC_CPG);
                sum_local_group.x += sum_per_channel_multi_group[channel_idx][src_thread_tile_y].x;
                sum_local_group.y += sum_per_channel_multi_group[channel_idx][src_thread_tile_y].y;
              }
            }
          }
        }
        static_assert(32 % THREADS_PER_GROUP == 0, "cannot shuffle");
        for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask >>= 1) {
          sum_local_group.x += __shfl_xor_sync(FINAL_MASK, sum_local_group.x, mask, 32);
          sum_local_group.y += __shfl_xor_sync(FINAL_MASK, sum_local_group.y, mask, 32);
        }
        if (threadIdx.x % THREADS_PER_GROUP == 0 && threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          if constexpr (USE_SHARED_RED_BUFFER) {
            static_assert(HARDWARE_CLUSTER || VIRTUAL_CLUSTER_SIZE == 1, "no distributed shared memory");
            if constexpr (STORE_MEAN_VAR_IN_SHARED_RED_BUFFER) {
              shared_red_buffer[threadIdx.x / THREADS_PER_GROUP] = compute_mean_var(sum_local_group);
            } else {
              shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + threadIdx.x / THREADS_PER_GROUP] = sum_local_group;
            }
          } else {
            *reinterpret_cast<float2*>(
                &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                             virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                             (threadIdx.x / THREADS_PER_GROUP) * virtual_cluster_dim_y + virtual_block_idx_y) *
                            2]) = sum_local_group;
          }
        }
      }

      virtual_cluster_sync<VIRTUAL_CLUSTER_SIZE, PERSISTENT, HARDWARE_CLUSTER>(barrier);

      // Group sum
      __shared__ float2 mean_var[MAX_NUM_GROUPS_PER_BLOCK];
      if constexpr (!STORE_MEAN_VAR_IN_SHARED_RED_BUFFER) {
        // The number of threads to calculate the sum of each group (should be a power of 2 for warp reduce)
        constexpr int THREADS_PER_GROUP = std::min(std::min(32U, round_up_pow2(virtual_cluster_dim_y)),
                                                   round_up_pow2(BLOCK_DIM_X / MAX_NUM_GROUPS_PER_BLOCK / 2 + 1));
        static_assert(BLOCK_DIM_X >= MAX_NUM_GROUPS_PER_BLOCK * THREADS_PER_GROUP, "not enough threads");
        float2 sum_global_group = {0.f, 0.f};
        if (threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          if constexpr (C_PER_BLOCK % CPG == 0) {
            // Special case: no cross-virtual_cluster_dim_x reduction
            float2 buffer[up_div(virtual_cluster_dim_y, THREADS_PER_GROUP)];
            for (int i = threadIdx.x % THREADS_PER_GROUP; i < virtual_cluster_dim_y; i += THREADS_PER_GROUP) {
              float2 val;
              if constexpr (USE_SHARED_RED_BUFFER) {
                if constexpr (VIRTUAL_CLUSTER_SIZE == 1) {
                  val = shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + threadIdx.x / THREADS_PER_GROUP];
                } else {
                  static_assert(HARDWARE_CLUSTER, "no distributed shared memory");
                  float2 const* src_shared_red_buffer = cg::this_cluster().map_shared_rank(
                      shared_red_buffer, i * virtual_cluster_dim_x + virtual_block_idx_x);
                  val = src_shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + threadIdx.x / THREADS_PER_GROUP];
                }
              } else {
                val = *reinterpret_cast<float2 const*>(
                    &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                                 virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                                 (threadIdx.x / THREADS_PER_GROUP) * virtual_cluster_dim_y + i) *
                                2]);
              }
              buffer[i / THREADS_PER_GROUP] = val;
            }
            for (int i = threadIdx.x % THREADS_PER_GROUP; i < virtual_cluster_dim_y; i += THREADS_PER_GROUP) {
              float2 val = buffer[i / THREADS_PER_GROUP];
              sum_global_group.x += val.x;
              sum_global_group.y += val.y;
            }
          } else {
            // Common case: cross-virtual_cluster_dim_x reduction
            int local_group_idx = block_group_start + threadIdx.x / THREADS_PER_GROUP;
            for (int i = threadIdx.x % THREADS_PER_GROUP; i < VIRTUAL_CLUSTER_SIZE; i += THREADS_PER_GROUP) {
              int src_virtual_block_idx_x = i % virtual_cluster_dim_x;
              int src_block_channel_start = src_virtual_block_idx_x * C_PER_BLOCK + c_loop * C_PER_CLUSTER;
              int src_block_group_start = src_block_channel_start / CPG;
              int relative_group_idx = local_group_idx - src_block_group_start;
              if (0 <= relative_group_idx && relative_group_idx < MAX_NUM_GROUPS_PER_BLOCK) {
                float2 val;
                if constexpr (USE_SHARED_RED_BUFFER) {
                  static_assert(HARDWARE_CLUSTER, "no distributed shared memory");
                  static_assert(VIRTUAL_CLUSTER_SIZE != 1,
                                "layout error: should not add (step * MAX_NUM_GROUPS_PER_BLOCK)");
                  float2 const* src_shared_red_buffer = cg::this_cluster().map_shared_rank(shared_red_buffer, i);
                  val = src_shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + relative_group_idx];
                } else {
                  val = *reinterpret_cast<float2 const*>(
                      &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                                   src_virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                                   relative_group_idx * virtual_cluster_dim_y + i / virtual_cluster_dim_x) *
                                  2]);
                }
                sum_global_group.x += val.x;
                sum_global_group.y += val.y;
              }
            }
          }
        }
        if constexpr (USE_SHARED_RED_BUFFER && VIRTUAL_CLUSTER_SIZE > 1) {
          // Need cluster sync after distributed shared memory access, otherwise behavior is undefined
          if constexpr (PERSISTENT) {
            if (nc_scheduler.at_end(n)) {
              cg::this_cluster().barrier_arrive();
            }
          } else {
            cg::this_cluster().barrier_arrive();
          }
        }
        static_assert(32 % THREADS_PER_GROUP == 0, "cannot shuffle");
        for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask >>= 1) {
          sum_global_group.x += __shfl_xor_sync(FINAL_MASK, sum_global_group.x, mask, 32);
          sum_global_group.y += __shfl_xor_sync(FINAL_MASK, sum_global_group.y, mask, 32);
        }
        if (threadIdx.x % THREADS_PER_GROUP == 0 && threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          mean_var[threadIdx.x / THREADS_PER_GROUP] = compute_mean_var(sum_global_group);
        }
        __syncthreads();
      }

      auto get_mean_var = [&](int relative_group_idx) {
        return STORE_MEAN_VAR_IN_SHARED_RED_BUFFER ? shared_red_buffer[relative_group_idx]
                                                   : mean_var[relative_group_idx];
      };

      if (mean_var_out) {
        static_assert(MAX_NUM_GROUPS_PER_BLOCK <= BLOCK_DIM_X, "need loop");
        if (virtual_block_idx_y == 0 && threadIdx.x < MAX_NUM_GROUPS_PER_BLOCK) {
          int g = block_group_start + threadIdx.x;
          if (C_PER_BLOCK % CPG == 0 || g < G) {
            *reinterpret_cast<float2*>(&mean_var_out[(n_loop * G + g) * 2]) = get_mean_var(threadIdx.x);
          }
        }
      }

      float frag_mean[VEC_ELEMS / GCD_VEC_CPG];
      float frag_var[VEC_ELEMS / GCD_VEC_CPG];
      for (int k = 0; k < VEC_ELEMS; k += GCD_VEC_CPG) {
        frag_mean[k / GCD_VEC_CPG] = get_mean_var((thread_channel_start + k) / CPG - block_group_start).x;
        frag_var[k / GCD_VEC_CPG] = get_mean_var((thread_channel_start + k) / CPG - block_group_start).y;
      }

      for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
        int64_t input_idx =
            n_loop * HW * C +
            (virtual_block_idx_y * ROWS_PER_BLOCK + j * ROWS_PER_IO + threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)) * C +
            thread_channel_start;
        U val;
        if constexpr (LOAD_TWICE) {
          val = *reinterpret_cast<U const*>(&x[input_idx]);
        } else {
          val = frag[j];
        }
        for (int k = 0; k < VEC_ELEMS; k++) {
          float f = ((float)val.data[k] - frag_mean[k / GCD_VEC_CPG]) * rsqrtf(frag_var[k / GCD_VEC_CPG] + eps) *
                        (float)uw.data[k] +
                    (float)ub.data[k];
          if constexpr (SILU) f = f / (1.f + expf(-f));
          val.data[k] = f;
        }
        *reinterpret_cast<U*>(&out[input_idx]) = val;
      }

      if constexpr (!STORE_MEAN_VAR_IN_SHARED_RED_BUFFER && USE_SHARED_RED_BUFFER && VIRTUAL_CLUSTER_SIZE > 1) {
        if constexpr (PERSISTENT) {
          if (nc_scheduler.at_end(n)) {
            cg::this_cluster().barrier_wait();
          }
        } else {
          cg::this_cluster().barrier_wait();
        }
      }

      if constexpr (!PERSISTENT) {
        break;
      }
      step ^= 1;
    }
  }
}

enum WgradSyncMethod {
  WGRAD_ARRIVE_AND_WAIT_GRID = 0,  // grid arrive after the last virtual cluster sync
  WGRAD_ARRIVE_AND_WAIT_GROUP,     // group arrive after the last virtual cluster sync (a group sync means synchronizing
                                   // all clusters cooperating on the same groups)
  WGRAD_REUSE_SUM_SYNC_GRID,       // grid sync together with the last virtual cluster sync
  WGRAD_REUSE_SUM_SYNC_GROUP,      // group sync together with the last virtual cluster sync
  WGRAD_SYNC_AT_LAST,              // add a sync at the end of NC loops
  WGRAD_SYNC_UNSPECIFIED,
};

template <typename T, int BLOCK_DIM_X, int BLOCKS_PER_SM, int G, int CPG, int HW, bool SILU, bool REQUIRES_WGRAD,
          int ROWS_PER_BLOCK, int C_PER_BLOCK, int C_PER_CLUSTER, int VEC_ELEMS, bool PERSISTENT,
          int NUM_VIRTUAL_CLUSTERS, bool LOAD_TWICE, bool HARDWARE_CLUSTER, WgradSyncMethod wgrad_sync_method,
          class CompileCondition = CompileConditionAlwaysTrue>
__global__ __launch_bounds__(BLOCK_DIM_X, BLOCKS_PER_SM) void gn_bwd_cuda_kernel(
    T* __restrict__ grad_input, T* __restrict__ grad_weight, T* __restrict__ grad_bias,
    T const* __restrict__ grad_output, T const* __restrict__ x, T const* __restrict__ w, T const* __restrict__ b,
    float const* __restrict__ mean_var, float eps, int64_t n, float* __restrict__ red_buffer,
    unsigned* __restrict__ barrier) {
  // Procedure Overview
  //   1. Thread sum: read from gmem, write partial sum to smem, store input in registers (if no LOAD_TWICE)
  //   2. Block sum: read from smem, write partial sum to gmem (or distributed shared memory if HARDWARE_CLUSTER is
  //   used),
  //        write wgrad to gmem at the last loop (at each loop if not CONSTANT_C_LOOP)
  //   3. Group sum: read from gmem, write mean&var to smem
  //   4. Scale: read mean&var from smem, read input from gmem (if LOAD_TWICE), write output to gmem
  //   5. Wgrad sum: read from gmem, write to gmem

  static_assert(BLOCK_DIM_X % 32 == 0, "warp shuffle error");

  constexpr int C = G * CPG;
  static_assert(C % C_PER_CLUSTER == 0, "cannot divide channels into clusters");
  static_assert(C_PER_CLUSTER % C_PER_BLOCK == 0, "cannot divide a cluster into blocks");
  static_assert(C_PER_CLUSTER % CPG == 0, "no reduce between clusters, would produce incorrect results");
  static_assert(!(C_PER_BLOCK % CPG == 0 && C_PER_CLUSTER != C_PER_BLOCK),
                "inefficient configuration, please reduce C_PER_CLUSTER");

  static_assert(ROWS_PER_BLOCK * C_PER_BLOCK % BLOCK_DIM_X == 0, "cannot divide tile into threads");
  struct alignas(VEC_ELEMS * sizeof(T)) U {
    T data[VEC_ELEMS];
  };

  // This function computes mean_dyw and mean_xdyw.
  // The function name is not changed because it has the same logic as the forward pass.
  auto compute_mean_var = [&](float2 sum) {
    float mean_dyw = sum.x / (HW * CPG);
    float mean_xdyw = sum.y / (HW * CPG);
    return float2{mean_dyw, mean_xdyw};
  };

  static_assert(HW % ROWS_PER_BLOCK == 0,
                "HW must be divisible by ROWS_PER_BLOCK to determine the number of blocks on the HW axis");
  constexpr int MAX_NUM_GROUPS_PER_BLOCK =
      C_PER_BLOCK % CPG == 0 ? C_PER_BLOCK / CPG : up_div(C_PER_BLOCK - gcd(C_PER_BLOCK, CPG), CPG) + 1;
  constexpr int VIRTUAL_CLUSTER_SIZE = (C_PER_CLUSTER / C_PER_BLOCK) * (HW / ROWS_PER_BLOCK);
  constexpr int virtual_cluster_dim_x = C_PER_CLUSTER / C_PER_BLOCK;
  constexpr int virtual_cluster_dim_y = HW / ROWS_PER_BLOCK;
  int virtual_block_idx_x = (blockIdx.x % VIRTUAL_CLUSTER_SIZE) % virtual_cluster_dim_x;
  int virtual_block_idx_y = (blockIdx.x % VIRTUAL_CLUSTER_SIZE) / virtual_cluster_dim_x;

  if constexpr (CompileCondition::matches()) {
    int step = 0;
    constexpr bool CONSTANT_C_LOOP = PERSISTENT && NUM_VIRTUAL_CLUSTERS % (C / C_PER_CLUSTER) == 0;
    if constexpr (!CONSTANT_C_LOOP) {
      static_assert(wgrad_sync_method != WGRAD_ARRIVE_AND_WAIT_GROUP && wgrad_sync_method != WGRAD_REUSE_SUM_SYNC_GROUP,
                    "grid sync is required when each block is responsible for multiple channel ranges");
    }
    NCScheduler<false, C, C_PER_CLUSTER, NUM_VIRTUAL_CLUSTERS, PERSISTENT> nc_scheduler(
        n);  // TODO: I don't know why the template specialization with CONSTANT_C_LOOP=true is slower.

    [[maybe_unused]] int virtual_cluster_idx_c = blockIdx.y % (C / C_PER_CLUSTER);
    [[maybe_unused]] cg::grid_group::arrival_token wgrad_sync_token;
    [[maybe_unused]] float dw_thread[VEC_ELEMS];
    [[maybe_unused]] float db_thread[VEC_ELEMS];
    [[maybe_unused]] __shared__ union {
      float2 dwdb_block_buffer[BLOCK_DIM_X][VEC_ELEMS];
      struct {
        float wgrad_buffer[BLOCK_DIM_X / 32][32];
        float bgrad_buffer[BLOCK_DIM_X / 32][32];
      } transpose_buffer;
    } union_smem;
    if constexpr (REQUIRES_WGRAD && CONSTANT_C_LOOP) {
      for (int i = 0; i < VEC_ELEMS; i++) {
        dw_thread[i] = 0.f;
        db_thread[i] = 0.f;
      }
    }
    float* red_buffer_wgrad =
        &red_buffer[(2 * NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK) * 2];
    unsigned* barrier_wgrad = barrier + NUM_VIRTUAL_CLUSTERS;
    if constexpr (REQUIRES_WGRAD && wgrad_sync_method != WGRAD_SYNC_AT_LAST) {
      if (nc_scheduler.at_end(n)) {
        static_assert(PERSISTENT, "persistent is a must for reducing wgrad");
        if constexpr (wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GRID) {
          wgrad_sync_token = group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE, PERSISTENT>(
              barrier_wgrad, blockIdx.x + blockIdx.y == 0);
        } else if constexpr (wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GROUP) {
          wgrad_sync_token =
              group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE / (C / C_PER_CLUSTER), PERSISTENT>(
                  barrier_wgrad + virtual_cluster_idx_c, blockIdx.x + blockIdx.y / (C / C_PER_CLUSTER) == 0);
        } else if constexpr (wgrad_sync_method == WGRAD_REUSE_SUM_SYNC_GRID) {
          wgrad_sync_token = group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE, PERSISTENT>(
              barrier_wgrad, blockIdx.x + blockIdx.y == 0);
          group_barrier_wait(barrier_wgrad, wgrad_sync_token);
        } else if constexpr (wgrad_sync_method == WGRAD_REUSE_SUM_SYNC_GROUP) {
          wgrad_sync_token =
              group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE / (C / C_PER_CLUSTER), PERSISTENT>(
                  barrier_wgrad + virtual_cluster_idx_c, blockIdx.x + blockIdx.y / (C / C_PER_CLUSTER) == 0);
          group_barrier_wait(barrier_wgrad + virtual_cluster_idx_c, wgrad_sync_token);
        }
      }
    }

    while (true) {  // TODO: unroll the loop
      if constexpr (PERSISTENT) {
        if (nc_scheduler.at_end(n)) {
          break;
        }
      }
      auto [n_loop, c_loop] = nc_scheduler.get_nc();
      if constexpr (PERSISTENT) {
        nc_scheduler.next(n);
      }
      static_assert(C_PER_BLOCK % VEC_ELEMS == 0, "cannot vectorize");
      static_assert((BLOCK_DIM_X * VEC_ELEMS) % C_PER_BLOCK == 0,
                    "each block should load one or more C_PER_BLOCK at once");
      constexpr int ROWS_PER_IO = BLOCK_DIM_X * VEC_ELEMS / C_PER_BLOCK;
      static_assert(ROWS_PER_BLOCK % ROWS_PER_IO == 0, "cannot determine the IO times per batch");
      int block_channel_start = virtual_block_idx_x * C_PER_BLOCK + c_loop * C_PER_CLUSTER;
      int block_group_start = block_channel_start / CPG;
      int thread_channel_start = block_channel_start + threadIdx.x % (C_PER_BLOCK / VEC_ELEMS) * VEC_ELEMS;
      U frag_x[ROWS_PER_BLOCK / ROWS_PER_IO];
      U frag_dy[ROWS_PER_BLOCK / ROWS_PER_IO];

      constexpr int GCD_VEC_CPG = gcd(VEC_ELEMS, CPG);

      constexpr bool SINGLE_GROUP_PER_BLOCK = CPG % C_PER_BLOCK == 0;
      [[maybe_unused]] __shared__ float2 sum_per_channel_multi_group[C_PER_BLOCK / GCD_VEC_CPG][relative_prime(
          128 / (int)sizeof(float2), ROWS_PER_IO)];
      [[maybe_unused]] __shared__ float2 sum_per_channel_single_group[BLOCK_DIM_X / 32];

      float frag_mean[VEC_ELEMS / GCD_VEC_CPG];
      float frag_var[VEC_ELEMS / GCD_VEC_CPG];
      for (int k = 0; k < VEC_ELEMS; k += GCD_VEC_CPG) {
        float2 value = *reinterpret_cast<float2 const*>(&mean_var[(n_loop * G + (thread_channel_start + k) / CPG) * 2]);
        frag_mean[k / GCD_VEC_CPG] = value.x;
        frag_var[k / GCD_VEC_CPG] = value.y;
      }

      U uw = *reinterpret_cast<U const*>(&w[thread_channel_start]);
      U ub;
      if constexpr (SILU) {
        ub = *reinterpret_cast<U const*>(&b[thread_channel_start]);
      }
      if constexpr (REQUIRES_WGRAD && !CONSTANT_C_LOOP) {
        for (int i = 0; i < VEC_ELEMS; i++) {
          dw_thread[i] = 0.f;
          db_thread[i] = 0.f;
        }
      }

      if constexpr (LOAD_TWICE) {
        float2 frag_sum_per_channel[VEC_ELEMS / GCD_VEC_CPG]{};
        for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
          int64_t input_idx =
              n_loop * HW * C +
              (virtual_block_idx_y * ROWS_PER_BLOCK + j * ROWS_PER_IO + threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)) * C +
              thread_channel_start;
          U ux = *reinterpret_cast<U const*>(&x[input_idx]);
          U udy = *reinterpret_cast<U const*>(&grad_output[input_idx]);
          for (int i = 0; i < VEC_ELEMS / GCD_VEC_CPG; i++) {
            float2 sum = frag_sum_per_channel[i];
            for (int k = 0; k < GCD_VEC_CPG; k++) {
              float rnorm = rsqrtf(frag_var[i] + eps);
              float x_norm =
                  ((float)ux.data[i * GCD_VEC_CPG + k] - frag_mean[i]) * rnorm;  // TODO: store rsqrtf in mean_var
              float grad_gn = udy.data[i * GCD_VEC_CPG + k];
              if constexpr (SILU) {
                float x_gn = x_norm * (float)uw.data[i * GCD_VEC_CPG + k] + (float)ub.data[i * GCD_VEC_CPG + k];
                float s = 1.f / (1.f + expf(-x_gn));
                grad_gn *= s * (1.f + x_gn * (1.f - s));
              }
              sum.x += grad_gn * (float)uw.data[i * GCD_VEC_CPG + k];
              sum.y += x_norm * (grad_gn * (float)uw.data[i * GCD_VEC_CPG + k]);
              if constexpr (REQUIRES_WGRAD) {
                dw_thread[i * GCD_VEC_CPG + k] += x_norm * grad_gn;
                db_thread[i * GCD_VEC_CPG + k] += grad_gn;
              }
            }
            frag_sum_per_channel[i] = sum;
          }
        }
        for (int i = 0; i < VEC_ELEMS / GCD_VEC_CPG; i++) {
          if constexpr (SINGLE_GROUP_PER_BLOCK) {
            for (int mask = 16; mask > 0; mask >>= 1) {
              frag_sum_per_channel[i].x += __shfl_xor_sync(FINAL_MASK, frag_sum_per_channel[i].x, mask, 32);
              frag_sum_per_channel[i].y += __shfl_xor_sync(FINAL_MASK, frag_sum_per_channel[i].y, mask, 32);
            }
            static_assert(VEC_ELEMS / GCD_VEC_CPG == 1, "process only one element for each warp");
            if (threadIdx.x % 32 == 0) {
              sum_per_channel_single_group[threadIdx.x / 32] = frag_sum_per_channel[i];
            }
          } else {
            sum_per_channel_multi_group[i * (C_PER_BLOCK / VEC_ELEMS) + threadIdx.x % (C_PER_BLOCK / VEC_ELEMS)]
                                       [threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)] = frag_sum_per_channel[i];
          }
        }
        __syncthreads();
      } else {
        for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
          int64_t input_idx =
              n_loop * HW * C +
              (virtual_block_idx_y * ROWS_PER_BLOCK + j * ROWS_PER_IO + threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)) * C +
              thread_channel_start;
          frag_x[j] = *reinterpret_cast<U const*>(&x[input_idx]);
          frag_dy[j] = *reinterpret_cast<U const*>(&grad_output[input_idx]);
        }

        for (int i = 0; i < VEC_ELEMS / GCD_VEC_CPG; i++) {
          float2 sum = {0.f, 0.f};
          for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
            for (int k = 0; k < GCD_VEC_CPG; k++) {
              float rnorm = rsqrtf(frag_var[i] + eps);
              float x_norm = ((float)frag_x[j].data[i * GCD_VEC_CPG + k] - frag_mean[i]) *
                             rnorm;  // TODO: store rsqrtf in mean_var
              float grad_gn = frag_dy[j].data[i * GCD_VEC_CPG + k];
              if constexpr (SILU) {
                float x_gn = x_norm * (float)uw.data[i * GCD_VEC_CPG + k] + (float)ub.data[i * GCD_VEC_CPG + k];
                float s = 1.f / (1.f + expf(-x_gn));
                grad_gn *= s * (1.f + x_gn * (1.f - s));
              }
              sum.x += grad_gn * (float)uw.data[i * GCD_VEC_CPG + k];
              sum.y += x_norm * (grad_gn * (float)uw.data[i * GCD_VEC_CPG + k]);
              if constexpr (REQUIRES_WGRAD) {
                dw_thread[i * GCD_VEC_CPG + k] += x_norm * grad_gn;
                db_thread[i * GCD_VEC_CPG + k] += grad_gn;
              }
            }
          }
          if constexpr (SINGLE_GROUP_PER_BLOCK) {
            for (int mask = 16; mask > 0; mask >>= 1) {
              sum.x += __shfl_xor_sync(FINAL_MASK, sum.x, mask, 32);
              sum.y += __shfl_xor_sync(FINAL_MASK, sum.y, mask, 32);
            }
            static_assert(VEC_ELEMS / GCD_VEC_CPG == 1, "process only one element for each warp");
            if (threadIdx.x % 32 == 0) {
              sum_per_channel_single_group[threadIdx.x / 32] = sum;
            }
          } else {
            sum_per_channel_multi_group[i * (C_PER_BLOCK / VEC_ELEMS) + threadIdx.x % (C_PER_BLOCK / VEC_ELEMS)]
                                       [threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)] = sum;
          }
        }
        __syncthreads();
      }

      if ((CONSTANT_C_LOOP && nc_scheduler.at_end(n)) || !CONSTANT_C_LOOP) {
        constexpr int NT_C = max_divisor(C_PER_BLOCK, BLOCK_DIM_X);  // Number of threads on the C axis
        constexpr int NT_R =
            1;  // std::min(32, (int)round_down_pow2(BLOCK_DIM_X / NT_C));  // Number of threads on the ROWS axis
        // TODO: swizzle for NT_R
        for (int i = 0; i < VEC_ELEMS; i++) {
          union_smem.dwdb_block_buffer[threadIdx.x][i ^ ((threadIdx.x / (16 / VEC_ELEMS)) & (VEC_ELEMS - 1))] =
              float2{dw_thread[i], db_thread[i]};
        }
        __syncthreads();
        static_assert(NT_C * NT_R <= BLOCK_DIM_X, "not enough threads");
        static_assert(C_PER_BLOCK % NT_C == 0, "need to loop once more and check c < C_PER_BLOCK");
        for (int i = 0; i < C_PER_BLOCK / NT_C; i++) {
          int c = i * NT_C + threadIdx.x / NT_R;
          float dw_block = 0.f;
          float db_block = 0.f;
          if (BLOCK_DIM_X == NT_C * NT_R || threadIdx.x < NT_C * NT_R) {
            for (int j = threadIdx.x % NT_R; j < ROWS_PER_IO; j += NT_R) {
              int src_thread = j * (C_PER_BLOCK / VEC_ELEMS) + c / VEC_ELEMS;
              float2 val = union_smem.dwdb_block_buffer[src_thread][(c % VEC_ELEMS) ^ ((src_thread / (16 / VEC_ELEMS)) &
                                                                                       (VEC_ELEMS - 1))];
              dw_block += val.x;
              db_block += val.y;
            }
          }
          static_assert(32 % NT_R == 0, "cannot shuffle");
          for (int mask = NT_R / 2; mask > 0; mask >>= 1) {
            dw_block += __shfl_xor_sync(FINAL_MASK, dw_block, mask, 32);
            db_block += __shfl_xor_sync(FINAL_MASK, db_block, mask, 32);
          }
          if (BLOCK_DIM_X == NT_C * NT_R || threadIdx.x < NT_C * NT_R) {
            if (threadIdx.x % NT_R == 0) {
              if constexpr (CONSTANT_C_LOOP) {
                *reinterpret_cast<float2*>(
                    &red_buffer_wgrad
                        [((blockIdx.y / (C / C_PER_CLUSTER) * virtual_cluster_dim_y + virtual_block_idx_y) * C +
                          c_loop * C_PER_CLUSTER + virtual_block_idx_x * C_PER_BLOCK + c) *
                         2]) = float2{dw_block, db_block};
              } else {
                *reinterpret_cast<float2*>(
                    &red_buffer_wgrad[((n_loop * virtual_cluster_dim_y + virtual_block_idx_y) * C +
                                       c_loop * C_PER_CLUSTER + virtual_block_idx_x * C_PER_BLOCK + c) *
                                      2]) = float2{dw_block, db_block};
              }
            }
          }
        }
      }

      constexpr bool USE_SHARED_RED_BUFFER = HARDWARE_CLUSTER || VIRTUAL_CLUSTER_SIZE == 1;
      constexpr bool STORE_MEAN_VAR_IN_SHARED_RED_BUFFER =
          VIRTUAL_CLUSTER_SIZE == 1 &&
          MAX_NUM_GROUPS_PER_BLOCK == 1;  // MAX_NUM_GROUPS_PER_BLOCK > 1 is possible but not implemented
      [[maybe_unused]] __align__(16)
          __shared__ float2 shared_red_buffer[MAX_NUM_GROUPS_PER_BLOCK * (STORE_MEAN_VAR_IN_SHARED_RED_BUFFER ? 1 : 2)];

      // Block sum
      if constexpr (SINGLE_GROUP_PER_BLOCK) {
        // block reduce
        if (threadIdx.x < 32) {
          float2 sum_local_group =
              threadIdx.x < BLOCK_DIM_X / 32 ? sum_per_channel_single_group[threadIdx.x] : float2{0.f, 0.f};
          constexpr int warp_num_pow2 = round_up_pow2(BLOCK_DIM_X / 32);
          for (int mask = warp_num_pow2 / 2; mask > 0; mask >>= 1) {
            sum_local_group.x += __shfl_xor_sync(FINAL_MASK, sum_local_group.x, mask, 32);
            sum_local_group.y += __shfl_xor_sync(FINAL_MASK, sum_local_group.y, mask, 32);
          }
          if (threadIdx.x == 0) {
            if constexpr (USE_SHARED_RED_BUFFER) {
              if constexpr (STORE_MEAN_VAR_IN_SHARED_RED_BUFFER) {
                shared_red_buffer[0] = compute_mean_var(sum_local_group);
              } else {
                shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + 0] = sum_local_group;
              }
            } else {
              *reinterpret_cast<float2*>(
                  &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                               virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                               // (threadIdx.x / THREADS_PER_GROUP) * virtual_cluster_dim_y +
                               virtual_block_idx_y) *
                              2]) = sum_local_group;
            }
          }
        }
      } else {
        // The number of threads to calculate the sum of each group (should be a power of 2 for warp reduce)
        constexpr int THREADS_PER_GROUP = std::min(std::min(32U, round_up_pow2(ROWS_PER_IO)),
                                                   round_up_pow2(BLOCK_DIM_X / MAX_NUM_GROUPS_PER_BLOCK / 2 + 1));
        static_assert(BLOCK_DIM_X >= MAX_NUM_GROUPS_PER_BLOCK * THREADS_PER_GROUP, "not enough threads");
        float2 sum_local_group = {0.f, 0.f};
        if (threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          int local_group_idx = block_group_start + threadIdx.x / THREADS_PER_GROUP;
          // TODO: map threads to both the CPG loop and the ROWS loop
          for (int local_c_loop = 0; local_c_loop < CPG; local_c_loop += GCD_VEC_CPG) {
            int c = local_group_idx * CPG + local_c_loop;
            if (C_PER_BLOCK % CPG == 0 || (c >= block_channel_start && c < block_channel_start + C_PER_BLOCK)) {
              for (int src_thread_tile_y = threadIdx.x % THREADS_PER_GROUP; src_thread_tile_y < ROWS_PER_IO;
                   src_thread_tile_y += THREADS_PER_GROUP) {
                int channel_idx = (c - block_channel_start) / GCD_VEC_CPG;
                channel_idx = channel_idx % (VEC_ELEMS / GCD_VEC_CPG) * (C_PER_BLOCK / VEC_ELEMS) +
                              channel_idx / (VEC_ELEMS / GCD_VEC_CPG);
                sum_local_group.x += sum_per_channel_multi_group[channel_idx][src_thread_tile_y].x;
                sum_local_group.y += sum_per_channel_multi_group[channel_idx][src_thread_tile_y].y;
              }
            }
          }
        }
        static_assert(32 % THREADS_PER_GROUP == 0, "cannot shuffle");
        for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask >>= 1) {
          sum_local_group.x += __shfl_xor_sync(FINAL_MASK, sum_local_group.x, mask, 32);
          sum_local_group.y += __shfl_xor_sync(FINAL_MASK, sum_local_group.y, mask, 32);
        }
        if (threadIdx.x % THREADS_PER_GROUP == 0 && threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          if constexpr (USE_SHARED_RED_BUFFER) {
            static_assert(HARDWARE_CLUSTER || VIRTUAL_CLUSTER_SIZE == 1, "no distributed shared memory");
            if constexpr (STORE_MEAN_VAR_IN_SHARED_RED_BUFFER) {
              shared_red_buffer[threadIdx.x / THREADS_PER_GROUP] = compute_mean_var(sum_local_group);
            } else {
              shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + threadIdx.x / THREADS_PER_GROUP] = sum_local_group;
            }
          } else {
            *reinterpret_cast<float2*>(
                &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                             virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                             (threadIdx.x / THREADS_PER_GROUP) * virtual_cluster_dim_y + virtual_block_idx_y) *
                            2]) = sum_local_group;
          }
        }
      }

      if constexpr (REQUIRES_WGRAD && wgrad_sync_method != WGRAD_SYNC_AT_LAST) {
        if (nc_scheduler.at_end(n)) {
          static_assert(PERSISTENT, "persistent is a must for reducing wgrad");
          if constexpr (wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GRID) {
            virtual_cluster_sync<VIRTUAL_CLUSTER_SIZE, PERSISTENT, HARDWARE_CLUSTER>(barrier);
            wgrad_sync_token = group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE, PERSISTENT>(
                barrier_wgrad, blockIdx.x + blockIdx.y == 0);
          } else if constexpr (wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GROUP) {
            virtual_cluster_sync<VIRTUAL_CLUSTER_SIZE, PERSISTENT, HARDWARE_CLUSTER>(barrier);
            wgrad_sync_token =
                group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE / (C / C_PER_CLUSTER), PERSISTENT>(
                    barrier_wgrad + virtual_cluster_idx_c, blockIdx.x + blockIdx.y / (C / C_PER_CLUSTER) == 0);
          } else if constexpr (wgrad_sync_method == WGRAD_REUSE_SUM_SYNC_GRID) {
            static_assert(!HARDWARE_CLUSTER,
                          "Distributed smem sync cannot reuse gmem sync. Use WGRAD_ARRIVE_AND_WAIT_GRID instead.");
            wgrad_sync_token = group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE, PERSISTENT>(
                barrier_wgrad, blockIdx.x + blockIdx.y == 0);
            group_barrier_wait(barrier_wgrad, wgrad_sync_token);
          } else if constexpr (wgrad_sync_method == WGRAD_REUSE_SUM_SYNC_GROUP) {
            static_assert(!HARDWARE_CLUSTER,
                          "Distributed smem sync cannot reuse gmem sync. Use WGRAD_ARRIVE_AND_WAIT_GROUP instead.");
            wgrad_sync_token =
                group_barrier_arrive<NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE / (C / C_PER_CLUSTER), PERSISTENT>(
                    barrier_wgrad + virtual_cluster_idx_c, blockIdx.x + blockIdx.y / (C / C_PER_CLUSTER) == 0);
            group_barrier_wait(barrier_wgrad + virtual_cluster_idx_c, wgrad_sync_token);
          }
        } else {
          virtual_cluster_sync<VIRTUAL_CLUSTER_SIZE, PERSISTENT, HARDWARE_CLUSTER>(barrier);
        }
      } else {
        virtual_cluster_sync<VIRTUAL_CLUSTER_SIZE, PERSISTENT, HARDWARE_CLUSTER>(barrier);
      }

      // Group sum
      __shared__ float2 mean_var[MAX_NUM_GROUPS_PER_BLOCK];
      if constexpr (!STORE_MEAN_VAR_IN_SHARED_RED_BUFFER) {
        // The number of threads to calculate the sum of each group (should be a power of 2 for warp reduce)
        constexpr int THREADS_PER_GROUP = std::min(std::min(32U, round_up_pow2(virtual_cluster_dim_y)),
                                                   round_up_pow2(BLOCK_DIM_X / MAX_NUM_GROUPS_PER_BLOCK / 2 + 1));
        static_assert(BLOCK_DIM_X >= MAX_NUM_GROUPS_PER_BLOCK * THREADS_PER_GROUP, "not enough threads");
        float2 sum_global_group = {0.f, 0.f};
        if (threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          if constexpr (C_PER_BLOCK % CPG == 0) {
            // Special case: no cross-virtual_cluster_dim_x reduction
            float2 buffer[up_div(virtual_cluster_dim_y, THREADS_PER_GROUP)];
            for (int i = threadIdx.x % THREADS_PER_GROUP; i < virtual_cluster_dim_y; i += THREADS_PER_GROUP) {
              float2 val;
              if constexpr (USE_SHARED_RED_BUFFER) {
                if constexpr (VIRTUAL_CLUSTER_SIZE == 1) {
                  val = shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + threadIdx.x / THREADS_PER_GROUP];
                } else {
                  static_assert(HARDWARE_CLUSTER, "no distributed shared memory");
                  float2 const* src_shared_red_buffer = cg::this_cluster().map_shared_rank(
                      shared_red_buffer, i * virtual_cluster_dim_x + virtual_block_idx_x);
                  val = src_shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + threadIdx.x / THREADS_PER_GROUP];
                }
              } else {
                val = *reinterpret_cast<float2 const*>(
                    &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                                 virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                                 (threadIdx.x / THREADS_PER_GROUP) * virtual_cluster_dim_y + i) *
                                2]);
              }
              buffer[i / THREADS_PER_GROUP] = val;
            }
            for (int i = threadIdx.x % THREADS_PER_GROUP; i < virtual_cluster_dim_y; i += THREADS_PER_GROUP) {
              float2 val = buffer[i / THREADS_PER_GROUP];
              sum_global_group.x += val.x;
              sum_global_group.y += val.y;
            }
          } else {
            // Common case: cross-virtual_cluster_dim_x reduction
            int local_group_idx = block_group_start + threadIdx.x / THREADS_PER_GROUP;
            for (int i = threadIdx.x % THREADS_PER_GROUP; i < VIRTUAL_CLUSTER_SIZE; i += THREADS_PER_GROUP) {
              int src_virtual_block_idx_x = i % virtual_cluster_dim_x;
              int src_block_channel_start = src_virtual_block_idx_x * C_PER_BLOCK + c_loop * C_PER_CLUSTER;
              int src_block_group_start = src_block_channel_start / CPG;
              int relative_group_idx = local_group_idx - src_block_group_start;
              if (0 <= relative_group_idx && relative_group_idx < MAX_NUM_GROUPS_PER_BLOCK) {
                float2 val;
                if constexpr (USE_SHARED_RED_BUFFER) {
                  static_assert(HARDWARE_CLUSTER, "no distributed shared memory");
                  static_assert(VIRTUAL_CLUSTER_SIZE != 1,
                                "layout error: should not add (step * MAX_NUM_GROUPS_PER_BLOCK)");
                  float2 const* src_shared_red_buffer = cg::this_cluster().map_shared_rank(shared_red_buffer, i);
                  val = src_shared_red_buffer[step * MAX_NUM_GROUPS_PER_BLOCK + relative_group_idx];
                } else {
                  val = *reinterpret_cast<float2 const*>(
                      &red_buffer[((step * gridDim.y + blockIdx.y) * VIRTUAL_CLUSTER_SIZE * MAX_NUM_GROUPS_PER_BLOCK +
                                   src_virtual_block_idx_x * virtual_cluster_dim_y * MAX_NUM_GROUPS_PER_BLOCK +
                                   relative_group_idx * virtual_cluster_dim_y + i / virtual_cluster_dim_x) *
                                  2]);
                }
                sum_global_group.x += val.x;
                sum_global_group.y += val.y;
              }
            }
          }
        }
        if constexpr (USE_SHARED_RED_BUFFER && VIRTUAL_CLUSTER_SIZE > 1) {
          // Need cluster sync after distributed shared memory access, otherwise behavior is undefined
          if constexpr (PERSISTENT) {
            if (nc_scheduler.at_end(n)) {
              cg::this_cluster().barrier_arrive();
            }
          } else {
            cg::this_cluster().barrier_arrive();
          }
        }
        static_assert(32 % THREADS_PER_GROUP == 0, "cannot shuffle");
        for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask >>= 1) {
          sum_global_group.x += __shfl_xor_sync(FINAL_MASK, sum_global_group.x, mask, 32);
          sum_global_group.y += __shfl_xor_sync(FINAL_MASK, sum_global_group.y, mask, 32);
        }
        if (threadIdx.x % THREADS_PER_GROUP == 0 && threadIdx.x / THREADS_PER_GROUP < MAX_NUM_GROUPS_PER_BLOCK) {
          mean_var[threadIdx.x / THREADS_PER_GROUP] = compute_mean_var(sum_global_group);
        }
        __syncthreads();
      }

      auto get_mean_var = [&](int relative_group_idx) {
        return STORE_MEAN_VAR_IN_SHARED_RED_BUFFER ? shared_red_buffer[relative_group_idx]
                                                   : mean_var[relative_group_idx];
      };

      float frag_dyw[VEC_ELEMS / GCD_VEC_CPG];
      float frag_xdyw[VEC_ELEMS / GCD_VEC_CPG];
      for (int k = 0; k < VEC_ELEMS; k += GCD_VEC_CPG) {
        frag_dyw[k / GCD_VEC_CPG] = get_mean_var((thread_channel_start + k) / CPG - block_group_start).x;
        frag_xdyw[k / GCD_VEC_CPG] = get_mean_var((thread_channel_start + k) / CPG - block_group_start).y;
      }

      for (int j = 0; j < ROWS_PER_BLOCK / ROWS_PER_IO; j++) {
        int64_t input_idx =
            n_loop * HW * C +
            (virtual_block_idx_y * ROWS_PER_BLOCK + j * ROWS_PER_IO + threadIdx.x / (C_PER_BLOCK / VEC_ELEMS)) * C +
            thread_channel_start;
        U ux;
        U udy;
        if constexpr (LOAD_TWICE) {
          ux = *reinterpret_cast<U const*>(&x[input_idx]);
          udy = *reinterpret_cast<U const*>(&grad_output[input_idx]);
        } else {
          ux = frag_x[j];
          udy = frag_dy[j];
        }
        U val;
        for (int k = 0; k < VEC_ELEMS; k++) {
          float rnorm = rsqrtf(frag_var[k / GCD_VEC_CPG] + eps);
          float x_norm = ((float)ux.data[k] - frag_mean[k / GCD_VEC_CPG]) * rnorm;  // TODO: store rsqrtf in mean_var
          float grad_gn = udy.data[k];
          if constexpr (SILU) {
            float x_gn = x_norm * (float)uw.data[k] + (float)ub.data[k];
            float s = 1.f / (1.f + expf(-x_gn));
            grad_gn *= s * (1.f + x_gn * (1.f - s));
          }
          val.data[k] =
              (grad_gn * (float)uw.data[k] - frag_dyw[k / GCD_VEC_CPG] - frag_xdyw[k / GCD_VEC_CPG] * x_norm) * rnorm;
        }
        *reinterpret_cast<U*>(&grad_input[input_idx]) = val;
      }

      if constexpr (!STORE_MEAN_VAR_IN_SHARED_RED_BUFFER && USE_SHARED_RED_BUFFER && VIRTUAL_CLUSTER_SIZE > 1) {
        if constexpr (PERSISTENT) {
          if (nc_scheduler.at_end(n)) {
            cg::this_cluster().barrier_wait();
          }
        } else {
          cg::this_cluster().barrier_wait();
        }
      }

      if constexpr (!PERSISTENT) {
        break;
      }
      step ^= 1;
    }

    // Wgrad sum
    if constexpr (REQUIRES_WGRAD) {
      static_assert(PERSISTENT, "cannot reduce wgrad");
      static_assert(C % 32 == 0, "cannot reduce wgrad");
      if constexpr (wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GRID) {
        group_barrier_wait(barrier_wgrad, wgrad_sync_token);
      } else if constexpr (wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GROUP) {
        group_barrier_wait(barrier_wgrad + virtual_cluster_idx_c, wgrad_sync_token);
      } else if constexpr (wgrad_sync_method == WGRAD_SYNC_AT_LAST) {
        cg::this_grid().sync();
      }

      // If group sync, map blocks that are responsible for the same range of channels to these channels (named "split
      // channels"); otherwise, map all blocks to all channels.
      constexpr bool split_channels =
          wgrad_sync_method == WGRAD_ARRIVE_AND_WAIT_GROUP || wgrad_sync_method == WGRAD_REUSE_SUM_SYNC_GROUP;

      for (int c = split_channels ? virtual_cluster_idx_c * C_PER_CLUSTER +
                                        32 * (blockIdx.y / (C / C_PER_CLUSTER) * VIRTUAL_CLUSTER_SIZE + blockIdx.x)
                                  : 32 * (blockIdx.y * VIRTUAL_CLUSTER_SIZE + blockIdx.x);
           split_channels ? c < (virtual_cluster_idx_c + 1) * C_PER_CLUSTER : c < C;
           c += split_channels ? 32 * (NUM_VIRTUAL_CLUSTERS / (C / C_PER_CLUSTER) * VIRTUAL_CLUSTER_SIZE)
                               : 32 * (NUM_VIRTUAL_CLUSTERS * VIRTUAL_CLUSTER_SIZE)) {
        int64_t rows = (CONSTANT_C_LOOP ? std::min(n, (int64_t)NUM_VIRTUAL_CLUSTERS / (C / C_PER_CLUSTER)) : n) *
                       virtual_cluster_dim_y;
        float sum_wgrad = 0.f;
        float sum_bgrad = 0.f;
        if ((split_channels &&
             (C_PER_CLUSTER % 32 == 0 || c + threadIdx.x % 32 < (virtual_cluster_idx_c + 1) * C_PER_CLUSTER)) ||
            (!split_channels && (C % 32 == 0 || c + threadIdx.x % 32 < C))) {
          for (int64_t i = threadIdx.x / 32; i < rows; i += BLOCK_DIM_X / 32) {
            float2 val = *reinterpret_cast<float2 const*>(&red_buffer_wgrad[(i * C + c + threadIdx.x % 32) * 2]);
            sum_wgrad += val.x;
            sum_bgrad += val.y;
          }
        }
        constexpr int warp_num_pow2 = round_up_pow2(BLOCK_DIM_X / 32);
        union_smem.transpose_buffer
            .wgrad_buffer[threadIdx.x / 32][(threadIdx.x % 32) ^ ((threadIdx.x / 32) * (32 / warp_num_pow2))] =
            sum_wgrad;
        union_smem.transpose_buffer
            .bgrad_buffer[threadIdx.x / 32][(threadIdx.x % 32) ^ ((threadIdx.x / 32) * (32 / warp_num_pow2))] =
            sum_bgrad;
        __syncthreads();
        for (int i = threadIdx.x / warp_num_pow2;
             i < 32 &&
             ((split_channels && (C_PER_CLUSTER % 32 == 0 || c + i < (virtual_cluster_idx_c + 1) * C_PER_CLUSTER)) ||
              (!split_channels && (C % 32 == 0 || c + i < C)));
             i += BLOCK_DIM_X / warp_num_pow2) {
          int j = threadIdx.x % warp_num_pow2;
          float sum_wgrad =
              j < BLOCK_DIM_X / 32 ? union_smem.transpose_buffer.wgrad_buffer[j][i ^ (j * (32 / warp_num_pow2))] : 0.f;
          float sum_bgrad =
              j < BLOCK_DIM_X / 32 ? union_smem.transpose_buffer.bgrad_buffer[j][i ^ (j * (32 / warp_num_pow2))] : 0.f;
          for (int mask = warp_num_pow2 / 2; mask > 0; mask >>= 1) {
            sum_wgrad += __shfl_xor_sync((uint64_t(1) << warp_num_pow2) - 1, sum_wgrad, mask, warp_num_pow2);
            sum_bgrad += __shfl_xor_sync((uint64_t(1) << warp_num_pow2) - 1, sum_bgrad, mask, warp_num_pow2);
          }
          if (j == 0) {
            grad_weight[c + i] = sum_wgrad;
            grad_bias[c + i] = sum_bgrad;
          }
        }
        __syncthreads();
      }
    }
  }
}

}  // namespace group_norm_v2
