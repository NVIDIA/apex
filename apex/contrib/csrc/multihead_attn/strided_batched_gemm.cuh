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

// symbol to be automatically resolved by PyTorch libs
/*
rocblas_datatype a_type       = rocblas_datatype_f16_r; // OK
rocblas_datatype b_type       = rocblas_datatype_f16_r; // OK
rocblas_datatype c_type       = rocblas_datatype_f16_r; // OK
rocblas_datatype d_type       = rocblas_datatype_f16_r;
rocblas_datatype compute_type       = rocblas_datatype_f32_r;

rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
int32_t           solution_index = 0;
rocblas_int       flags          = 0;
*/

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

// needed to work around calling rocblas API instead of hipblas API
static rocblas_operation hipOperationToRocOperation(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return rocblas_operation_none;
    case HIPBLAS_OP_T:
        return rocblas_operation_transpose;
    case HIPBLAS_OP_C:
        return rocblas_operation_conjugate_transpose;
    }
    AT_ERROR("HIPBLAS_STATUS_INVALID_ENUM");
}

static hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
    case rocblas_status_continue:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
    case rocblas_status_perf_degraded:
    case rocblas_status_check_numerics_fail:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    AT_ERROR("HIPBLAS_STATUS_INVALID_ENUM");
}

void RocblasStridedBatchedGemm(char transa, char transb, long m, long n, long k,
                    float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                    float beta, half *c, long ldc, long strideC, half *d, long ldd, long strideD, long batchCount, rocblas_gemm_algo algo, rocblas_int flags) {
    cublasOperation_t opa = convertTransToCublasOperation(transa);
    cublasOperation_t opb = convertTransToCublasOperation(transb);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);
    float fAlpha = alpha;
    float fBeta = beta;
    //THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                     hipOperationToRocOperation(opa), hipOperationToRocOperation(opb), (int)m, (int)n, (int)k,
                                     (void*)&fAlpha, a, rocblas_datatype_f16_r /*a_type*/, (int)lda, strideA,
                                     b, rocblas_datatype_f16_r /*b_type*/, (int)ldb, strideB,
                                     (void*)&fBeta, c, rocblas_datatype_f16_r /*c_type*/, (int)ldc, strideC,
                                     d, rocblas_datatype_f16_r /*d_type*/, int(ldd), strideD,
                                     (int)batchCount, rocblas_datatype_f32_r /*compute_type*/, algo, 0 /*solution_index*/, flags)));
}

void gemm_switch_fp32accum(char transa, char transb, long m, long n, long k,
                           float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                           float beta, half *c, long ldc, long strideC, half *d, long ldd, long strideD, long batchCount, rocblas_int flags) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  if        ( (transa == 't') && (transb == 'n') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, rocblas_gemm_algo_standard, flags); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, rocblas_gemm_algo_standard, flags); }
  } else if ( (transa == 'n') && (transb == 'n') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, rocblas_gemm_algo_standard, flags); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, rocblas_gemm_algo_standard, flags); }
  } else if ( (transa == 'n') && (transb == 't') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, rocblas_gemm_algo_standard, flags); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, rocblas_gemm_algo_standard, flags); }
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
                        b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, 0 /*flags*/);
}

} // namespace
