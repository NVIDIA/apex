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

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/fast_math.h>
#include <cutlass/pitch_linear_coord.h>

namespace {
cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't')
    return CUBLAS_OP_T;
  else if (trans == 'n')
    return CUBLAS_OP_N;
  else if (trans == 'c')
    return CUBLAS_OP_C;
  else {
    TORCH_CHECK(false, "trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void CublasStridedBatchedGemm(
    char transa, char transb, long m, long n, long k,
    float alpha, const half *a, long lda, long strideA, const half *b, long ldb,
    long strideB, float beta, half *c, long ldc, long strideC, long batchCount,
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);
  float fAlpha = alpha;
  float fBeta = beta;
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, opa, opb, (int)m, (int)n, (int)k, (void *)&fAlpha, a, CUDA_R_16F,
      (int)lda, strideA, b, CUDA_R_16F, (int)ldb, strideB, (void *)&fBeta, c,
      CUDA_R_16F, (int)ldc, strideC, (int)batchCount, CUDA_R_32F, algo));
}

} // namespace

// TODO(mkozuki): Make use of the int template parameters or discard them.
template <typename LayoutA, typename LayoutB, int SRC_A, int SRC_B, int DST_C>
void CutlassGemm_FP32Accum(
  cudaStream_t stream,
  long m, long n, long k, float alpha,
  const half* a, long lda, long long int batch_stride_A,
  const half* b, long ldb, long long int batch_stride_B,
  float beta,
  half* c, long ldc, long long int batch_stride_C, long batch_count
) {
  using Gemm = cutlass::gemm::device::GemmBatched<
    /* Element type of A matrix */half, /* Layout of A matrix */LayoutA,
    /* Element type of B matrix */half, /* Layout of B matrix */LayoutB,
    /* Element type of C matrix */half, /* Layout of C matrix */cutlass::layout::ColumnMajor,
    /* Element Accumulator*/float
  >;
  Gemm gemm_op;
  cutlass::Status status = gemm_op({
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
      {a, lda}, batch_stride_A,
      {b, ldb}, batch_stride_B,
      {c, ldc}, batch_stride_C,
      {c, ldc}, batch_stride_C,
      {alpha, beta}, static_cast<int>(batch_count)
  }, nullptr, stream);
  C10_CUDA_CHECK(status != cutlass::Status::kSuccess ? cudaErrorUnknown : cudaSuccess);
}

namespace {
void gemm_switch_fp32accum(char transa, char transb, long m,
                           long n, long k, float alpha, const half *a, long lda,
                           long strideA, const half *b, long ldb, long strideB,
                           float beta, half *c, long ldc, long strideC,
                           long batchCount) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  // printf("GEMM   -> %c%c M: %i N: %i K: %i Alpha: %f Beta: %f\n", (transa ==
  // 't' ? 'T' : 'N'), (transb =='t' ? 'T' : 'N'), m, n, k, alpha, beta);
  if ((transa == 't') && (transb == 'n')) {
    if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CublasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda,
                               strideA, b, ldb, strideB, beta, c, ldc, strideC,
                               batchCount, CUBLAS_GEMM_ALGO0_TENSOR_OP);
    }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 8, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 8, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 4, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 8, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::RowMajor,
                            cutlass::layout::ColumnMajor, 2, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else {
      CublasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda,
                               strideA, b, ldb, strideB, beta, c, ldc, strideC,
                               batchCount);
    }
  } else if ((transa == 'n') && (transb == 'n')) {
    if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CublasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda,
                               strideA, b, ldb, strideB, beta, c, ldc, strideC,
                               batchCount, CUBLAS_GEMM_ALGO0_TENSOR_OP);
    }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 8, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 8, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 4, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 8, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor, 2, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else {
      CublasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda,
                               strideA, b, ldb, strideB, beta, c, ldc, strideC,
                               batchCount);
    }
  } else if ((transa == 'n') && (transb == 't')) {
    if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CublasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda,
                               strideA, b, ldb, strideB, beta, c, ldc, strideC,
                               batchCount, CUBLAS_GEMM_ALGO0_TENSOR_OP);
    }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 8, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 8, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 4, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 8, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 8, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 8, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 4, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 4, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 4, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 2, 8>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 2, 4>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<cutlass::layout::ColumnMajor,
                            cutlass::layout::RowMajor, 2, 2, 2>(
          stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c,
          ldc, strideC, batchCount);
    } else {
      CublasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda,
                               strideA, b, ldb, strideB, beta, c, ldc, strideC,
                               batchCount);
    }
  } else {
    TORCH_CHECK(false, "TransA and TransB are invalid");
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
                         long batchCount) {
  if ((m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX) ||
      (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX))

  {
    TORCH_CHECK(false, "Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, "
             "batchCount"
             "with the bound [val] <= %d",
             INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  gemm_switch_fp32accum(transa, transb, m, n, k, alpha, a, lda, strideA,
                        b, ldb, strideB, beta, c, ldc, strideC, batchCount);
}

} // namespace
