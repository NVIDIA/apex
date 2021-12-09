#pragma once

#include "stdint.h"
#include "stddef.h"
#include "cuda.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *fmhalib_get_error();

// qkv_ptr is a FP16 Tensor with shape [total, 3, num_heads, head_size] 
// cu_seqlens is a FP32 Tensor with shape [batch_size + 1] 
// ctx_ptr is a FP16 Tensor with shape [total, num_heads, head_size] 
// s_ptr is a FP16 Tensor with shape [batch_size, num_heads, max_seq_len, max_seq_len] 
// where max_seq_len should be any of [128, 256, 384, 512]  
void fmhalib_fwd(const void *qkv_ptr,
                 const int total,
                 const int num_heads,
                 const int head_size,
                 const int *cu_seqlens_ptr,
                 const int batch_size,
                 const float p_dropout,
                 const int max_seq_len,
                 const bool is_training,
                 const uint64_t rnd_seed,
                 const int64_t *offset_ptr,
                 const uint64_t rnd_offset,
                 bool is_device_rnd,
                 cudaStream_t stream,
                 void *ctx_ptr,
                 void *s_ptr);

#ifdef __cplusplus
}
#endif
