#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
// includes cublaslt
#include <cublasLt.h>
#endif
// FP64 Wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    double* A,
    int lda,
    double* B,
    int ldb,
    const float* beta,
    double* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_64F,
      lda,
      B,
      CUDA_R_64F,
      ldb,
      beta,
      C,
      CUDA_R_64F,
      ldc,
      CUDA_R_64F,
      CUBLAS_GEMM_DEFAULT);
}

// FP32 Wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    float* A,
    int lda,
    float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_32F,
      lda,
      B,
      CUDA_R_32F,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float* beta,
    at::Half* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      beta,
      C,
      CUDA_R_16F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600


int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float *beta, /* host pointer */
    at::Half* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BIAS;
  } 

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}







int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    double* A,
    int lda,
    double* B,
    int ldb,
    const float *beta, /* host pointer */
    double* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  return 1;
}

int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta, /* host pointer */
    float *C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_32F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }

  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          &heuristicResult.algo,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}



int gemm_bias_gelu_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float *beta, /* host pointer */
    at::Half* C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* gelu_in,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_AUX;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in));
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc));

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
  } 

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}


int gemm_bias_gelu_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    double* A,
    int lda,
    double* B,
    int ldb,
    const float *beta, /* host pointer */
    double* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void *gelu_in,
    const void* bias) {
  return 1;
}


int gemm_bias_gelu_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta, /* host pointer */
    float *C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* gelu_in,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_AUX;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in));
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc));

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
  } 

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_32F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}



int gemm_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float *beta, /* host pointer */
    at::Half* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bgrad) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
  } 

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}







int gemm_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    double* A,
    int lda,
    double* B,
    int ldb,
    const float *beta, /* host pointer */
    double* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bgrad) {
  return 1;
}

int gemm_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta, /* host pointer */
    float *C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bgrad) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_32F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }

  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          &heuristicResult.algo,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}


int gemm_dgelu_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float *beta, /* host pointer */
    at::Half* C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    const void *gelu_in,
    const void *bgrad) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc));

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

int gemm_dgelu_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    double *A,
    int lda,
    double *B,
    int ldb,
    const float *beta, /* host pointer */
    double *C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    const void *gelu_in,
    const void *bgrad) {
    return 1;
}

int gemm_dgelu_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta, /* host pointer */
    float *C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    const void *gelu_in,
    const void *bgrad) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc));

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_32F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

#endif

template <typename T>
int linear_bias_forward_cuda(at::Tensor input, T *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    const float beta_one       = 1.0;
    int status = 1;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    status = gemm_bias_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_T,
    CUBLAS_OP_N,
    out_features,
    batch_size,
    in_features,
    &alpha, /* host pointer */
    weight,
    in_features,
    input.data_ptr<T>(),
    in_features,
    &beta_zero, /* host pointer */
    output.data_ptr<T>(),
    out_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    static_cast<const void*>(bias.data_ptr<T>()));
#endif
    if (status != 0){
        output.copy_(bias);
        status = gemm_bias(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          out_features,
          batch_size,
          in_features,
          &alpha,
          weight,
          in_features,
          input.data_ptr<T>(),
          in_features,
          &beta_one,
          output.data_ptr<T>(),
          out_features);
    }
    return status;
}

    
template <typename T>
int linear_bias_backward_cuda(T *input, T *weight, T *d_output, int in_features, int batch_size, int out_features, T *d_weight, T *d_bias, T *d_input,  void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    int status = 1;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    status = gemm_bgradb_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    in_features,
    out_features,
    batch_size,
    &alpha, /* host pointer */
    input,
    in_features,
    d_output,
    out_features,
    &beta_zero, /* host pointer */
    d_weight,
    in_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    static_cast<const void*>(d_bias));
#endif
    

    if (status != 0){
    
        status = gemm_bias(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          in_features,
          out_features,
          batch_size,
          &alpha,
          input,
          in_features,
          d_output,
          out_features,
          &beta_zero,
          d_weight,
          in_features);
    }
    
    status = gemm_bias(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      in_features,
      batch_size,
      out_features,
      &alpha,
      weight,
      in_features,
      d_output,
      out_features,
      &beta_zero,
      d_input,
      in_features);
    return status;

}

template <typename T>
int linear_gelu_linear_forward_cuda(T *input, T *weight1, T *bias1, T *weight2, T *bias2, int in_features, int hidden_features, int batch_size, int out_features, T *output1, T *output2, T *gelu_in, void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    int status = 1;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    status = gemm_bias_gelu_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_T,
    CUBLAS_OP_N,
    hidden_features,
    batch_size,
    in_features,
    &alpha, /* host pointer */
    weight1,
    in_features,
    input,
    in_features,
    &beta_zero, /* host pointer */
    output1,
    hidden_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    static_cast<const void*>(gelu_in),
    static_cast<const void*>(bias1));
    status = gemm_bias_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_T,
    CUBLAS_OP_N,
    out_features,
    batch_size,
    hidden_features,
    &alpha, /* host pointer */
    weight2,
    hidden_features,
    output1,
    hidden_features,
    &beta_zero, /* host pointer */
    output2,
    out_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    static_cast<const void*>(bias2));
    return status;
#else 
    return 1;
#endif
}

template <typename T>
int linear_gelu_linear_backward_cuda(T *input, T *gelu_in, T *output1, T *weight1, T *weight2, T *d_output1, T *d_output2, int in_features, int batch_size, int hidden_features, int out_features, T *d_weight1, T *d_weight2, T *d_bias1, T *d_bias2, T *d_input, void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    int status = 1;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
//wgrad for first gemm
    status = gemm_bgradb_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    hidden_features,
    out_features,
    batch_size,
    &alpha, /* host pointer */
    output1,
    hidden_features,
    d_output2,
    out_features,
    &beta_zero, /* host pointer */
    d_weight2,
    hidden_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    static_cast<const void*>(d_bias2));
//dgrad for second GEMM
    status = gemm_dgelu_bgradb_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    hidden_features,
    batch_size,
    out_features,
    &alpha, /* host pointer */
    weight2,
    hidden_features,
    d_output2,
    out_features,
    &beta_zero, /* host pointer */
    d_output1,
    hidden_features,
    lt_workspace,
    1 << 22,
    stream,
    static_cast<const void*>(gelu_in),
    static_cast<const void*>(d_bias1));
//wgrad for the first GEMM
    status = gemm_bias(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      in_features,
      hidden_features,
      batch_size,
      &alpha,
      input,
      in_features,
      d_output1,
      hidden_features,
      &beta_zero,
      d_weight1,
      in_features);

//dgrad for the first GEMM
    status = gemm_bias(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      in_features,
      batch_size,
      hidden_features,
      &alpha,
      weight1,
      in_features,
      d_output1,
      hidden_features,
      &beta_zero,
      d_input,
      in_features);
#endif
    return status;

}


template int linear_bias_forward_cuda<at::Half>(at::Tensor input, at::Half *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template int linear_bias_forward_cuda<float>(at::Tensor input, float *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template int linear_bias_forward_cuda<double>(at::Tensor input, double *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template int linear_bias_backward_cuda<at::Half>(at::Half *input, at::Half *weight, at::Half *d_output, int in_features, int batch_size, int out_features, at::Half *d_weight, at::Half *d_bias, at::Half *d_input,  void *lt_workspace) ;

template int linear_bias_backward_cuda<float>(float *input, float *weight, float *d_output, int in_features, int batch_size, int out_features, float *d_weight, float *d_bias, float *d_input,  void *lt_workspace) ;

template int linear_bias_backward_cuda<double>(double *input, double *weight, double *d_output, int in_features, int batch_size, int out_features, double *d_weight, double *d_bias, double *d_input,  void *lt_workspace) ;


template int linear_gelu_linear_forward_cuda<at::Half>(at::Half *input, at::Half *weight1, at::Half *bias1, at::Half *weight2, at::Half *bias2, int in_features, int hidden_features, int batch_size, int out_features, at::Half *output1, at::Half *output2, at::Half *gelu_in, void *lt_workspace) ;

template int linear_gelu_linear_forward_cuda<float>(float *input, float *weight1, float *bias1, float *weight2, float *bias2, int in_features, int hidden_features, int batch_size, int out_features, float *output1, float *output2, float *gelu_in, void *lt_workspace);

template int linear_gelu_linear_forward_cuda<double>(double *input, double *weight1, double *bias1, double *weight2, double *bias2, int in_features, int hidden_features, int batch_size, int out_features, double *output1, double *output2, double *gelu_in, void *lt_workspace) ;

template int linear_gelu_linear_backward_cuda<at::Half>(at::Half *input, at::Half *gelu_in, at::Half *output1, at::Half *weight1, at::Half *weight2, at::Half *d_output1, at::Half *d_output2, int in_features, int batch_size, int hidden_features, int out_features, at::Half *d_weight1, at::Half *d_weight2, at::Half *d_bias1, at::Half *d_bias2, at::Half *d_input, void *lt_workspace);

template int linear_gelu_linear_backward_cuda<float>(float *input, float *gelu_in, float *output1, float *weight1, float *weight2, float *d_output1, float *d_output2, int in_features, int batch_size, int hidden_features, int out_features, float *d_weight1, float *d_weight2, float *d_bias1, float *d_bias2, float *d_input, void *lt_workspace);

template int linear_gelu_linear_backward_cuda<double>(double *input, double *gelu_in, double *output1, double *weight1, double *weight2, double *d_output1, double *d_output2, int in_features, int batch_size, int hidden_features, int out_features, double *d_weight1, double *d_weight2, double *d_bias1, double *d_bias2, double *d_input, void *lt_workspace);
