#pragma once

#include "stdint.h"
#include "stddef.h"
#include "cuda.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * All functions of the FMHALIB does not return the error flag and
 * does not throw any exception. If users want to check whether there
 * is any error, users can call this method after the FMHALIB function
 * is called.
 *
 * If there is no error, it would return nullptr.
 * If there is any error, it would return the error message.
 *
 * Note that if the empty string "" is returned, there is an error without
 * any error message instead of no errors. 
 *
 * Note that the error message is thread local, do not get the error message
 * in another thread.
 */
const char *fmhalib_error();

/**
 * This function may return -1 if seq_len is invalid.
 */
int fmhalib_seq_len(const int seq_len);


/**
 * qkv_ptr: FP16 Tensor with shape [total, 3, num_heads, head_size] 
 * cu_seqlens_ptr: INT32 Tensor with shape [batch_size + 1]
 * total: the total seqence length (not including padding) of the mini-batch   
 * num_heads: head number 
 * head_size: must be 64
 * batch_size: batch size
 * p_dropout: dropout probability
 * max_seq_len: must be <= 512
 * is_training: whether to run train or inference
 * is_nl: whether to run the nl kernel  
 * zero_tensors: whether to zero output tensors before applying forward  
 * seed_inc_func: the seed increment function pointer.   
 * stream: the CUDA stream.
 *
 * ctx_ptr: output FP16 tensor with shape [total, num_heads, head_size]
 * s_ptr: output FP16 Tensor with shape [batch_size, num_heads, seq_len, seq_len], where seq_len can be obtained by calling `fmhalib_seq_len(max_seq_len)` 
 */
void fmhalib_fwd(const void *qkv_ptr,
                 const void *cu_seqlens_ptr,
                 const int total,
                 const int num_heads,
                 const int head_size,
                 const int batch_size,
                 const float p_dropout,
                 const int max_seq_len,
                 const bool is_training,
                 const bool is_nl,
                 const bool zero_tensors,
                 void (*seed_inc_func)(uint64_t, uint64_t *, const int64_t **, uint64_t*, bool *),
                 cudaStream_t stream,
                 void *ctx_ptr,
                 void *s_ptr);

/**
 * dout_ptr: the gradient of the output `ctx_ptr` in fmhalib_fwd
 * qkv_ptr: same with the fmhalib_fwd 
 * cu_seqlens_ptr: same with the fmhalib_fwd
 * total: same with the fmhalib_fwd
 * num_heads: same with the fmhalib_fwd
 * head_size: same with the fmhalib_fwd
 * batch_size: same with the fmhalib_fwd
 * p_dropout: same with the fmhalib_fwd
 * max_seq_len: same with the fmhalib_fwd
 * is_nl: same with the fmhalib_fwd 
 * zero_tensors: same with the fmhalib_fwd 
 * stream: the CUDA stream
 *
 * softmax_ptr: the output `s_ptr` in fmhalib_fwd. Note that it may be overwritten inside the function!   
 * dqkv_ptr: the gradient of the input `qkv_ptr` in fmhalib_fwd
 * workspace: the workspace pointer
 * workspace_size: the workspace size. If dqkv_ptr == nullptr, no computation would be performed, and
 *                 the workspace size would be returned.
 */
void fmhalib_bwd(const void *dout_ptr,
                 const void *qkv_ptr,
                 const void *cu_seqlens_ptr,
                 const int total,
                 const int num_heads,
                 const int head_size,
                 const int batch_size,
                 const float p_dropout, 
                 const int max_seq_len,
                 const bool is_nl,
                 const bool zero_tensors,
                 cudaStream_t stream,
                 void *softmax_ptr,  // will be overwritten
                 void *dqkv_ptr,
                 void *workspace,
                 uint64_t *workspace_size);
#ifdef __cplusplus
}
#endif
