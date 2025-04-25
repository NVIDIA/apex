#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

/* Includes, cuda */

#include <cuda_runtime.h>
#include "type_shim.h"

/* Includes, blaslt */
#include <cublasLt.h>

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(error)                    \
    if(error != cudaSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Cuda error: '%s'(%d) at %s:%d\n", \
                cudaGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_CUBLASLT_ERROR
#define CHECK_CUBLASLT_ERROR(error)                                                      \
    if(error != CUBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "cudaBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

// BF16 Tensor core wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasLtHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int batch_count,
    float& alpha,
    float& beta,
    at::BFloat16* A,
        at::BFloat16* B,
    float*    C,
    float*    D,
    void*     d_workspace,
    int64_t   max_workspace_size,
    cudaStream_t   stream) {

    cublasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matA, CUDA_R_16BF, m, k, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matB, CUDA_R_16BF, n, k, n));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matC, CUDA_R_32F, m, n, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matD, CUDA_R_32F, m, n, m));

    cublasLtMatmulDesc_t matmul;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT_ERROR(
        cublasLtMatmulPreferenceSetAttribute(pref,
                                              CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 1;
    cublasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    uint64_t workspace_size = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);

    CHECK_CUBLASLT_ERROR(cublasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          A,
                                          matA,
                                          B,
                                          matB,
                                          &beta,
                                          C,
                                          matC,
                                          D,
                                          matD,
                                          &heuristicResult[0].algo,
                                          d_workspace,
                                          workspace_size,
                                          stream));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matA));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matB));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matC));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matD));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescDestroy(matmul));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceDestroy(pref));
    return;
}

// FP16 Tensor core wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasLtHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int batch_count,
    float& alpha,
    float& beta,
    at::Half* A,
    at::Half* B,
    float*    C,
    float*    D,
    void*     d_workspace,
    int64_t   max_workspace_size,
    cudaStream_t   stream) {
    cublasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matA, CUDA_R_16F, m, k, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matB, CUDA_R_16F, n, k, n));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matC, CUDA_R_32F, m, n, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matD, CUDA_R_32F, m, n, m));

    cublasLtMatmulDesc_t matmul;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT_ERROR(
        cublasLtMatmulPreferenceSetAttribute(pref,
                                              CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 1;
    cublasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    uint64_t workspace_size = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);

    CHECK_CUBLASLT_ERROR(cublasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          A,
                                          matA,
                                          B,
                                          matB,
                                          &beta,
                                          C,
                                          matC,
                                          D,
                                          matD,
                                          &heuristicResult[0].algo,
                                          d_workspace,
                                          workspace_size,
                                          stream));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matA));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matB));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matC));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matD));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescDestroy(matmul));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceDestroy(pref));
    return;
}


// FP32 wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasLtHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int batch_count,
    float& alpha,
    float& beta,
    float* A,
    float* B,
    float* C,
    float* D,
    void*   d_workspace,
    int64_t  max_workspace_size,
    cudaStream_t   stream) {
    cublasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matA, CUDA_R_32F, m, k, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matB, CUDA_R_32F, n, k, n));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matC, CUDA_R_32F, m, n, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matD, CUDA_R_32F, m, n, m));

    cublasLtMatmulDesc_t matmul;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        matmul, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT_ERROR(
        cublasLtMatmulPreferenceSetAttribute(pref,
                                              CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 1;
    cublasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    uint64_t workspace_size = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);

    CHECK_CUBLASLT_ERROR(cublasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          A,
                                          matA,
                                          B,
                                          matB,
                                          &beta,
                                          C,
                                          matC,
                                          D,
                                          matD,
                                          &heuristicResult[0].algo,
                                          d_workspace,
                                          workspace_size,
                                          stream));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matA));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matB));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matC));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matD));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescDestroy(matmul));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceDestroy(pref));
    return;
}

template <typename T>
void wgrad_gemm_accum_fp32_cuda(T *input, T *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim) {

    cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0;
    float beta  = 1.0;
    const int batch_count = 1;
    void*   d_workspace = nullptr;
    int64_t max_workspace_size = 32*1024*1024;
    if(max_workspace_size > 0) {
        at::Tensor workspace = at::empty({max_workspace_size}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
        d_workspace = workspace.data_ptr();
    }
    gemmex_wrapper(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        in_dim,         //m
        out_dim,        //n
        hidden_dim,     //k
        batch_count,
        alpha,
        beta,
        input,          //da
        d_output,       //db
        d_weight,       //dc
        d_weight,       //dd
        d_workspace,
        max_workspace_size,
        stream);
}

template void wgrad_gemm_accum_fp32_cuda<at::Half>(at::Half *input, at::Half *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp32_cuda<at::BFloat16>(at::BFloat16 *input, at::BFloat16 *d_output,  float *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp32_cuda<float>(float *input, float *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);


void wgrad_gemm_accum_fp32_cuda_stub(
  at::Tensor &input,
  at::Tensor &d_output,
  at::Tensor &d_weight) 
{
    at::Tensor input_2d, d_output_2d;
    // input tensor: collapse to the first dim
    auto in_sizes = input.sizes();
    if (input.dim() > 2) {
        input_2d = input.view({-1, in_sizes[in_sizes.size() - 1]});
    } else {
        input_2d = input;
    }
    // d_output tensor: collapse to the first dim
    auto d_out_sizes = d_output.sizes();
    if (d_output.dim() > 2) {
        d_output_2d = d_output.view({-1, d_out_sizes[d_out_sizes.size() - 1]});
    } else {
        d_output_2d = d_output;
    }

    const int hidden_dim = input_2d.size(0);  //k
    const int in_dim = input_2d.size(1);      //m
    const int out_dim = d_weight.size(0);     //n

    DISPATCH_FLOAT_HALF_AND_BFLOAT(input_2d.scalar_type(), 0, "wgrad_gemm_accum_fp32",
        wgrad_gemm_accum_fp32_cuda<scalar_t_0>(
            input_2d.data_ptr<scalar_t_0>(),
            d_output_2d.data_ptr<scalar_t_0>(),
            d_weight.data_ptr<float>(),
            in_dim,
            hidden_dim,
            out_dim);
    );
}
