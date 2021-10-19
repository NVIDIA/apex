#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

rocblas_datatype a_type       = rocblas_datatype_f16_r;
rocblas_datatype b_type       = rocblas_datatype_f16_r;
rocblas_datatype c_type       = rocblas_datatype_f16_r;
rocblas_datatype d_type       = rocblas_datatype_f16_r;
rocblas_datatype compute_type       = rocblas_datatype_f32_r;

rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
int32_t           solution_index = 0;
rocblas_int       flags          = 0;


cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    THError("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void RocblasStridedBatchedGemm(THCState *state, char transa, char transb, long m, long n, long k,
                    float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                    float beta, half *c, long ldc, long strideC, half *d, long ldd, long strideD, long batchCount, rocblas_gemm_algo algo) {
    cublasOperation_t opa = convertTransToCublasOperation(transa);
    cublasOperation_t opb = convertTransToCublasOperation(transb);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);
    float fAlpha = alpha;
    float fBeta = beta;
    THCublasCheck(rocblas_gemm_strided_batched_ex(handle,
                                     opa, opb, (int)m, (int)n, (int)k,
                                     (void*)&fAlpha, a, a_type, (int)lda, strideA,
                                     b, b_type, (int)ldb, strideB,
                                     (void*)&fBeta, c, c_type, (int)ldc, strideC,
				     d, d_type, int(ldd), strideD,
                                     (int)batchCount, compute_type, algo, solution_index, flags));
}

void gemm_switch_fp32accum(THCState *state, char transa, char transb, long m, long n, long k,
                           float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                           float beta, half *c, long ldc, long strideC, half *d, long ldd, long strideD, long batchCount) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  if        ( (transa == 't') && (transb == 'n') ) { 
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
    else                                                   { RocblasStridedBatchedGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
  } else if ( (transa == 'n') && (transb == 'n') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
    else                                                   { RocblasStridedBatchedGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
  } else if ( (transa == 'n') && (transb == 't') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {RocblasStridedBatchedGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
    else                                                   { RocblasStridedBatchedGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
  } else {
    AT_ASSERTM(false, "TransA and TransB are invalid");
  }
}

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t *lda, int64_t *ldb, int64_t *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  if(n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if(transa_)
  {
    if(m <= 1)
      *lda = std::max<int64_t>(k, 1);
  }
  else
  {
    if(k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if(transb_)
  {
    if(k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  }
  else
  {
    if(n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }

}

void HgemmStridedBatched(THCState *state, char transa, char transb, long m, long n, long k,
                             float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                             float beta, half *c, long ldc, long strideC, half *d, long ldd, long strideD, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )

  {
    THError("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  //gemm_switch(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
  gemm_switch_fp32accum(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount);
}

/******
at::Tensor strided_batched_gemm_cuda(
    float beta,
    at::Tensor in_result,
    float alpha,
    at::Tensor batch1,
    at::Tensor batch2) {

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  int64_t lda, ldb, ldc;
  at::Tensor result, input1, input2;
  if (in_result.stride(1) == 1)
  {
    transpose_result = false;
    result = in_result;
    ldc = result.stride(2);
  }
  else if (in_result.stride(2) == 1)
  {
    transpose_result = true;

    at::Tensor swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result = in_result;
    ldc = result.stride(1);
  } else { 
    AT_ASSERTM(false, "result should be contiguous");
  }

  if (batch1.stride(transpose_result ? 2 : 1) == 1 &&
      batch1.stride(transpose_result ? 1 : 2) != 0) {
    transpose_batch1 = 'n';
    input1 = batch1;
    lda = input1.stride(transpose_result ? 1 : 2);
  } else if (batch1.stride(transpose_result ? 1 : 2) == 1 &&
             batch1.stride(transpose_result ? 2 : 1) != 0) {
    transpose_batch1 = 't';
    input1 = batch1;
    lda = input1.stride(transpose_result ? 2 : 1);
  } else {
    AT_ASSERTM(false, "input1 should be contiguous");
  }

  if (batch2.stride(transpose_result ? 2 : 1) == 1 &&
      batch2.stride(transpose_result ? 1 : 2) != 0) {
    transpose_batch2 = 'n';
    input2 = batch2;
    ldb = input2.stride(transpose_result ? 1 : 2);
  } else if (batch2.stride(transpose_result ? 1 : 2) == 1 &&
             batch2.stride(transpose_result ? 2 : 1) != 0) {
    transpose_batch2 = 't';
    input2 = batch2;
    ldb = input2.stride(transpose_result ? 2 : 1);
  } else {
    AT_ASSERTM(false, "input2 should be contiguous");
  }
  int64_t num_batches = result.size(0);

  HgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result.size(transpose_result ? 2 : 1),
      result.size(transpose_result ? 1 : 2),
      input1.size(transpose_result ? 1 : 2),
      alpha,
      static_cast<const half*>(input1.data_ptr()), lda, input1.stride(0),
      static_cast<const half*>(input2.data_ptr()), ldb, input2.stride(0),
      beta,
      static_cast<half*>(result.data_ptr()), ldc, result.stride(0),
      num_batches);

  return in_result;
}

***/
