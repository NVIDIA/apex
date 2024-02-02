#pragma once
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
//#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <rocblas/rocblas.h>

//#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

//#include "cutlass/cutlass.h"
//#include "cutlass/gemm/gemm.h"
//#include "cutlass/gemm/wmma_gemm_traits.h"

#include "type_shim.h"

namespace {
cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't')
    return CUBLAS_OP_T;
  else if (trans == 'n')
    return CUBLAS_OP_N;
  else if (trans == 'c')
    return CUBLAS_OP_C;
  else {
    AT_ERROR("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void RocblasStridedBatchedGemm(char transa, char transb, long m, long n, long k,
                    float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                    float beta, half *c, long ldc, long strideC, long batchCount, hipblasGemmAlgo_t algo) {
    cublasOperation_t opa = convertTransToCublasOperation(transa);
    cublasOperation_t opb = convertTransToCublasOperation(transb);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);
    float fAlpha = alpha;
    float fBeta = beta;

    TORCH_CUDABLAS_CHECK(hipblasGemmStridedBatchedEx(
			    handle,
                            opa, 
			    opb, 
			    (int)m, 
			    (int)n, 
			    (int)k,
                            (void*)&fAlpha, 
			    a, 
			    HIP_R_16F /*a_type*/, 
			    (int)lda, 
			    strideA,
                            b, 
			    HIP_R_16F /*b_type*/, 
			    (int)ldb, 
			    strideB,
                            (void*)&fBeta, 
			    c, 
			    HIP_R_16F /*c_type*/, 
			    (int)ldc, 
			    strideC,
                            (int)batchCount,
			    HIPBLAS_COMPUTE_32F, 
			    algo));
}

void gemm_switch_fp32accum(char transa, char transb, long m, long n, long k,
                           float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                           float beta, half *c, long ldc, long strideC, long batchCount) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  if        ( (transa == 't') && (transb == 'n') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount, HIPBLAS_GEMM_DEFAULT); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount, HIPBLAS_GEMM_DEFAULT); }
  } else if ( (transa == 'n') && (transb == 'n') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount, HIPBLAS_GEMM_DEFAULT); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount, HIPBLAS_GEMM_DEFAULT); }
  } else if ( (transa == 'n') && (transb == 't') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC,  batchCount, HIPBLAS_GEMM_DEFAULT); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount, HIPBLAS_GEMM_DEFAULT); }
  } else {
    AT_ASSERTM(false, "TransA and TransB are invalid");
  }
}

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k,
                    int64_t *lda, int64_t *ldb, int64_t *ldc) {
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0 and at
  // least as big the result requires (even if the value won't be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}

void HgemmStridedBatched(char transa, char transb, long m,
                         long n, long k, float alpha, const half *a, long lda,
                         long strideA, const half *b, long ldb, long strideB,
                         float beta, half *c, long ldc, long strideC,
                         half *d, long ldd, long strideD, long batchCount) {

  if ((m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX) ||
      (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX))

  {
    AT_ERROR("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, "
             "batchCount"
             "with the bound [val] <= %d",
             INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  // gemm_switch_fp32accum(transa, transb, m, n, k, alpha, a, lda, strideA,
  //                       b, ldb, strideB, beta, c, ldc, strideC, batchCount);
  gemm_switch_fp32accum(transa, transb, m, n, k, alpha, a, lda, strideA, 
                        b, ldb, strideB, beta, c, ldc, strideC, batchCount);
}

} // namespace
