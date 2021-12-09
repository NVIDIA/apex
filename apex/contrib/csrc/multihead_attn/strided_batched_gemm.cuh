#pragma once
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

//#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"

cublasOperation_t convertTransToCublasOperation(char trans);

void CublasStridedBatchedGemm(
    char transa, char transb, long m, long n, long k,
    float alpha, const half *a, long lda, long strideA, const half *b, long ldb,
    long strideB, float beta, half *c, long ldc, long strideC, long batchCount,
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP);

template <cutlass::MatrixLayout::Kind A_LAYOUT,
          cutlass::MatrixLayout::Kind B_LAYOUT, int SRC_A, int SRC_B, int DST_C>
void CutlassGemm_FP32Accum(cudaStream_t stream, long m, long n, long k,
                           float alpha, const half *a, long lda, long strideA,
                           const half *b, long ldb, long strideB, float beta,
                           half *c, long ldc, long strideC, long batchCount);

void gemm_switch_fp32accum(char transa, char transb, long m,
                           long n, long k, float alpha, const half *a, long lda,
                           long strideA, const half *b, long ldb, long strideB,
                           float beta, half *c, long ldc, long strideC,
                           long batchCount);

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k,
                    int64_t *lda, int64_t *ldb, int64_t *ldc);

void HgemmStridedBatched(char transa, char transb, long m,
                         long n, long k, float alpha, const half *a, long lda,
                         long strideA, const half *b, long ldb, long strideB,
                         float beta, half *c, long ldc, long strideC,
                         long batchCount);
