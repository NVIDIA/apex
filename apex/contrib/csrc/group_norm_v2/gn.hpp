#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace group_norm_v2 {

struct Meta {
  int64_t red_buffer_size;
  int64_t barrier_size;
  int BLOCK_DIM_X;
  int C_PER_BLOCK;
  int ROWS_PER_BLOCK;
  int VEC_ELEMS;
  bool LOAD_TWICE;
  int BLOCKS_PER_SM;
  bool HARDWARE_CLUSTER;
  int wgrad_sync_method;
};

template <typename T>
void gn_cuda(T* out, T* x, T* w, T* b, float eps, bool silu, int64_t n, int64_t hw, int num_groups,
             int channels_per_group, float* mean_var_out, float* red_buffer, unsigned* barrier, int sm_margin,
             cudaStream_t stream, int device_id, Meta* meta_ptr, bool meta_only);

template <typename T>
void gn_bwd_cuda(T* grad_input, T* grad_weight, T* grad_bias, T* grad_output, T* x, T* w, T* b, float* mean_var,
                 float eps, bool silu, int64_t n, int64_t hw, int num_groups, int channels_per_group, float* red_buffer,
                 unsigned* barrier, int sm_margin, cudaStream_t stream, int device_id, Meta* meta_ptr, bool meta_only);

}  // namespace group_norm_v2
