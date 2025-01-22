
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <torch/torch.h>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#define DEBUG 0

#include "type_shim.h"

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                \
  if (error != hipSuccess)                    \
  {                                           \
    fprintf(stderr,                           \
            "Hip error: '%s'(%d) at %s:%d\n", \
            hipGetErrorString(error),         \
            error,                            \
            __FILE__,                         \
            __LINE__);                        \
    exit(EXIT_FAILURE);                       \
  }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                  \
  if (error != HIPBLAS_STATUS_SUCCESS)                                                \
  {                                                                                   \
    fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
    fprintf(stderr, "\n");                                                            \
    exit(EXIT_FAILURE);                                                               \
  }
#endif

#define DISPATCH_TYPES(TYPE, NAME, ...)             \
  switch (TYPE)                                     \
  {                                                 \
  case at::ScalarType::Half:                        \
  {                                                 \
    constexpr auto compute_t = CUBLAS_COMPUTE_32F;  \
    constexpr auto compute_datatype_t = CUDA_R_32F; \
    constexpr auto datatype_t = CUDA_R_16F;         \
    using scalar_t = at::Half;                      \
    __VA_ARGS__();                                  \
    break;                                          \
  }                                                 \
  case at::ScalarType::BFloat16:                    \
  {                                                 \
    constexpr auto compute_t = CUBLAS_COMPUTE_32F;  \
    constexpr auto compute_datatype_t = CUDA_R_32F; \
    constexpr auto datatype_t = CUDA_R_16BF;        \
    using scalar_t = at::BFloat16;                  \
    __VA_ARGS__();                                  \
    break;                                          \
  }                                                 \
  case at::ScalarType::Float:                       \
  {                                                 \
    constexpr auto compute_t = CUBLAS_COMPUTE_32F;  \
    constexpr auto compute_datatype_t = CUDA_R_32F; \
    constexpr auto datatype_t = CUDA_R_32F;         \
    using scalar_t = float;                         \
    __VA_ARGS__();                                  \
    break;                                          \
  }                                                 \
  case at::ScalarType::Double:                      \
  {                                                 \
    constexpr auto compute_t = CUBLAS_COMPUTE_64F;  \
    constexpr auto compute_datatype_t = CUDA_R_64F; \
    constexpr auto datatype_t = CUDA_R_64F;         \
    using scalar_t = double;                        \
    __VA_ARGS__();                                  \
    break;                                          \
  }                                                 \
  default:                                          \
    AT_ERROR(#NAME, " not implemented type ");      \
  }


hipDataType get_dtype(at::Tensor A)
{
  hipDataType dataType;

  if (A.scalar_type() == at::ScalarType::BFloat16)
  {
    dataType = HIP_R_16F;
  }
  if (A.scalar_type() == at::ScalarType::Half)
  {
    dataType = HIP_R_16F;
  }
  if (A.scalar_type() == at::ScalarType::Float)
  {
    dataType = HIP_R_32F;
  }
  if (A.scalar_type() == at::ScalarType::Double)
  {
    dataType = HIP_R_64F;
  }
  // The E4M3 is mainly used for the weights, and the E5M2 is for the gradient.
  if (A.scalar_type() == at::ScalarType::Float8_e5m2fnuz)
  {
    dataType = HIP_R_8F_E5M2_FNUZ;
  }
  if (A.scalar_type() == at::ScalarType::Float8_e4m3fnuz)
  {
    dataType = HIP_R_8F_E4M3_FNUZ;
  }

  return dataType;
}


/********************************************************************************************************************************************************
 *
 * D = Epilogue{  (alpha_s * (A * B) +  beta_s * C) +  bias_v } * scaleD_v
 *
 ******************************************************************************************************************************************************/
int gemm_lt(
    hipblasOperation_t trans_a,
    hipblasOperation_t trans_b,
    const float *alpha,
    const float *beta,
    at::Tensor A,
    at::Tensor B,
    at::Tensor C,
    at::Tensor bias,
    at::Tensor gelu,
    bool use_bias,
    bool use_grad,
    bool use_gelu)
{

  hipStream_t stream;
  hipblasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  hipblasGetStream(handle, &stream);

#if DEBUG
  std::cout << "gemm_lt " << std::endl;
#endif
  if ((trans_a == HIPBLAS_OP_T) && (trans_b == HIPBLAS_OP_T))
  {
    std::cout << "Both Transose is not supported";
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

  /* ============================================================================================
   *   Matrix layout:
   *  1. Set the Data type of matrix elements.
   *  3. Set the layout: Size/shape of the matrix. This depends if transpose is needed or not.
   *  4. Set the leading dimentions
   *  
   */
  hipblasLtMatrixLayout_t matA = nullptr, matB = nullptr, matC = nullptr;

  hipDataType dtype_a = get_dtype(A);
  hipDataType dtype_b = get_dtype(B);
  hipDataType dtype_c = get_dtype(C);

  int64_t m = trans_a == HIPBLAS_OP_T ? A.size(0) : A.size(1);
  int64_t k = trans_a == HIPBLAS_OP_T ? A.size(1) : A.size(0);
  int64_t n = trans_b == HIPBLAS_OP_T ? B.size(1) : B.size(0);

  int64_t lda = 0, ldb = 0, ldd = 0;

  if ((trans_a == HIPBLAS_OP_T) && (trans_b != HIPBLAS_OP_T))
  {
    lda = k;
    ldb = k;
  } // TN
  else if ((trans_a != HIPBLAS_OP_T) && (trans_b == HIPBLAS_OP_T))
  {
    lda = m;
    ldb = n;
  } // NT
  else if ((trans_a != HIPBLAS_OP_T) && (trans_b != HIPBLAS_OP_T))
  {
    lda = m;
    ldb = k;
  } // NN

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype_a, trans_a == HIPBLAS_OP_T ? k : m, 
                                                                    trans_a == HIPBLAS_OP_T ? m : k, lda));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype_b, trans_b == HIPBLAS_OP_T ? n : k, 
                                                                    trans_b == HIPBLAS_OP_T ? k : n, ldb));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dtype_c, m, n, m));

  /* ============================================================================================
   *  Matmul desc:
   * 1. Create operation descriptor with compute data type
   * 2. Set transpose operation  
   */
  hipblasLtMatmulDesc_t matmulDesc = nullptr;

  hipblasComputeType_t desc_computeType = HIPBLAS_COMPUTE_32F;
  hipDataType desc_dataType = HIP_R_32F;

  if (A.scalar_type() == at::ScalarType::Double)
  {
    desc_computeType = HIPBLAS_COMPUTE_64F;
    desc_dataType = HIP_R_64F;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmulDesc, desc_computeType, desc_dataType));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, 
                                                        &trans_a, sizeof(trans_a)));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, 
                                                        &trans_b, sizeof(trans_b)));

  /* ============================================================================================
   *   Configure epilogue
   * 1. Set mat-mul post-ops: bias, bgradb, gelu.  
   * 2. 
   */

  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

  hipDataType dtype_bias = get_dtype(bias);
  hipDataType dtype_gelu = get_dtype(gelu);

  auto d_bias = static_cast<void *>(bias.data_ptr());
  auto d_gelu = static_cast<void *>(gelu.data_ptr());
  int64_t ld_gelu = (int64_t)gelu.size(0);

  if (use_bias && use_gelu)
  {
    if (use_grad)
    {
      epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
    }
    else
    {
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, 
                                                          &d_bias, sizeof(d_bias)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, 
                                                          &dtype_bias, sizeof(dtype_bias)));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, 
                                                          &d_gelu, sizeof(d_gelu)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, 
                                                          &ld_gelu, sizeof(ld_gelu)));
    // CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, 
    //                                                       &dtype_gelu, sizeof(dtype_gelu)));
  }
  else if (use_bias)
  {
    if (use_grad)
    {
      epilogue = HIPBLASLT_EPILOGUE_BGRADB;
    }
    else
    {
      epilogue = HIPBLASLT_EPILOGUE_BIAS;
    }
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, 
                                                          &d_bias, sizeof(d_bias)));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, 
                                                          &dtype_bias, sizeof(dtype_bias)));
  }
  else if (use_gelu)
  {
    if (use_grad)
    {
      epilogue = HIPBLASLT_EPILOGUE_DGELU;
    }
    else
    {
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;
    }
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, 
                                                          &d_gelu, sizeof(d_gelu)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, 
                                                          &ld_gelu, sizeof(ld_gelu)));
    // CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, 
    //                                                       &dtype_gelu, sizeof(dtype_gelu)));
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, 
                                                        &epilogue, sizeof(epilogue)));

  /* ============================================================================================
   *   Algo Get Heuristic
   * 1. retrieves the possible algorithms for given input matrices A, B and C, and the output matrix D.
   *    decription/layout. In our case matrux C and D are same. search result is in heuristicResultsArray[].
   */
  hipblasLtMatmulPreference_t pref;

  const int request_solutions = 1;
  int       returnedAlgoCount = 0;
  uint64_t     workspace_size = 0;
  void             *workspace = nullptr;

  hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, matA, matB, matC, matC, 
                                                        pref, request_solutions, heuristicResult, 
                                                        &returnedAlgoCount));

  if (returnedAlgoCount == 0)
  {
    std::cerr << "No valid solution found!" << std::endl;
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }

  for (int i = 0; i < returnedAlgoCount; i++)
  {
    workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
  }

  hipMalloc(&workspace, workspace_size);
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                              &workspace, sizeof(workspace_size)));

  /* ============================================================================================
   * Matmul
   */
  const void *d_a = static_cast<const void *>(A.data_ptr());
  const void *d_b = static_cast<const void *>(B.data_ptr());
  void       *d_c = static_cast<void *>(C.data_ptr());

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmulDesc, alpha, d_a, matA,
                                        d_b, matB, beta, static_cast<const void *>(d_c), 
                                        matC, d_c, matC, &heuristicResult[0].algo,
                                        workspace, workspace_size, stream));

#if DEBUG
  std::cout << "\nTensor-A:\n" << A 
            << "\nTensor-B:\n" << B 
            << "\nTensor-C:\n" << C 
            << "\nTensor-Bias:\n" << bias << std::endl;
  std::cout << "\nSizes: A[" << A.size(0) << "," << A.size(1) << "]" << std::endl;
  std::cout << "\nSizes: B[" << B.size(0) << "," << B.size(1) << "]" << std::endl;
  std::cout << "\nSizes: C[" << C.size(0) << "," << C.size(1) << "]" << std::endl;
  std::cout << "\nValues:: m:" << m << ", k:" << k << ", n:" << n << std::endl;
  std::cout << "lda: " << lda << "\tldb: " << ldb << "\tldd: " << ldd << "\tm: " << m << "\tk: " << k << "\tn: " << n << std::endl;
#endif

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmulDesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

  return HIPBLAS_STATUS_SUCCESS;
}


template <hipblasComputeType_t ComputeType, typename TensorType, hipDataType DataType>
hipblasStatus_t gemm_bias( hipblasOperation_t transa, hipblasOperation_t transb,
                           int64_t m, int64_t n, int64_t k, const float *alpha, const float *beta,
                           const TensorType *A, const TensorType *B, TensorType *C)
{
  hipblasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  int64_t lda = n;
  int64_t ldb = k;
  int64_t ldc = m;

#if DEBUG
  std::cout << "gemm_bias " << std::endl;
#endif
  return hipblasGemmEx(handle, transa, transb, m,  n,   k,  alpha,  A,  DataType,   lda,  B,  DataType,  
                       ldb,  beta,  C,  DataType,  ldc,  ComputeType,  CUBLAS_GEMM_DEFAULT);
}


/****************************************************************************
 * output[batch_size, out_features] = input[batch_size, in_features] * weight[out_features,in_features] + bias[out_features]
 ****************************************************************************/
at::Tensor linear_bias_forward(at::Tensor input, at::Tensor weight, at::Tensor bias)
{
  const float alpha = 1.0, beta = 0.0;

  int64_t  batch_size      = input.size(0); // input[batch_size, in_features]
  int64_t  in_features     = input.size(1);
  int64_t  out_features    = weight.size(0); // weight[out_features,in_features]

  at::Tensor dummy_gelu = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

  // ********************************************************************************** 
  // output[batch_size, out_features] = input[batch_size, in_features] * weight[out_features,in_features] + bias[out_features]
  // **********************************************************************************
  auto output = at::zeros({batch_size, out_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
#if DEBUG
  std::cout << "linear_bias_forward " << std::endl;
#endif
  if (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt) {
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_T, HIPBLAS_OP_N, &alpha, &beta, weight, input, output, bias, dummy_gelu, true, false, false));
  } else {
    DISPATCH_TYPES(input.scalar_type(), "linear_bias_forward", [&] {
    auto result = gemm_bias<compute_t, scalar_t, datatype_t>(
                            HIPBLAS_OP_T, HIPBLAS_OP_N, out_features, batch_size, in_features, 
                            &alpha, &beta, weight.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
    if (result != 0) { fprintf(stderr, "INVALID RESULT for linear_bias_forward\n"); }
    });
  }

  return {output};
}

/****************************************************************************
 * In the backward pass, we compute the gradients of the loss with respect to input, weight, and bias.
 * The key matrix operations are:
 *  1. Gradient of Input  : grad_input[batch_size, in_features]  = output[batch_size, out_features] * weight[out_features,in_features] 
 *  2. Gradient of Weights: grad_weight[out_features,in_features] = input[batch_size, in_features]  * output[batch_size, out_features]
 *  3. Gradient of Bias   : grad_bias=sum(dY)
 **************************************************************************/
std::vector<at::Tensor> linear_bias_backward(at::Tensor input, at::Tensor weight, at::Tensor output)
{
  const float alpha = 1.0, beta = 0.0;

  int64_t  batch_size   = input.size(0); // input[batch_size, in_features]
  int64_t  in_features  = input.size(1);
  int64_t  out_features = weight.size(0); // weight[out_features,in_features]

  auto grad_bias  = at::zeros(out_features, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto dummy_gelu = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto grad_weight = at::zeros({out_features,in_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto grad_input = at::zeros({batch_size, in_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  
#if DEBUG
  std::cout << "linear_bias_backward " << std::endl;
#endif
  if (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt) {
  // **********************************************************************************
  // Gradient of Input  :
  // grad_input [batch_size, in_features] = output[batch_size, out_features] * Weight[out_features,in_features]
  // **********************************************************************************
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_N, HIPBLAS_OP_N, &alpha, &beta, weight, output, grad_input, grad_bias, dummy_gelu, false, false, false));

  // **********************************************************************************
  // Gradient of Weights:
  // grad_weight[out_features,in_features] = input[batch_size, in_features](T)  * output[batch_size, out_features] 
  // **********************************************************************************
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_N, HIPBLAS_OP_T, &alpha, &beta, output, input, grad_weight, grad_bias, dummy_gelu, true, false, false));

  // **********************************************************************************
  // ToDo: Check why HipBLASLt fail to get bgrad above so this step is not needed.
  // db=sum(dY)
  // **********************************************************************************
    grad_bias = output.sum(0, false);
  } else {
    DISPATCH_TYPES(input.scalar_type(), "linear_bias_forward", [&] {
    auto result = gemm_bias<compute_t, scalar_t, datatype_t>(
                            HIPBLAS_OP_N, HIPBLAS_OP_T, in_features, out_features, batch_size, 
                            &alpha, &beta, input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), grad_weight.data_ptr<scalar_t>());
    if (result != 0) { fprintf(stderr, "INVALID RESULT for linear_bias_forward\n"); }
    });

    DISPATCH_TYPES(input.scalar_type(), "linear_bias_forward", [&] {
    auto result = gemm_bias<compute_t, scalar_t, datatype_t>(
                            HIPBLAS_OP_N, HIPBLAS_OP_N, in_features, batch_size, out_features,
                            &alpha, &beta, weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>());
    if (result != 0) { fprintf(stderr, "INVALID RESULT for linear_bias_forward\n"); }
    });
  }
  return {grad_input, grad_weight, grad_bias};
}

/****************************************************************************
 *
 * [Linear] https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
 * [GELU]   https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
 *
 *   module combines dense layers with GELU activations in a single neural network layer.
 *   layer consists of two dense sub-layers, each followed by a GELU activation function. 
 *   It takes an input tensor and passes it through these sub-layers to produce the final output.
 *
 *   layer consists of the following internal layers:
 *      dense1:     The first dense layer.  
 *                         output[batch_size, hidden_features] = input[batch_size, in_features] * weight[hidden_features,in_features] + bias[hidden_features]
 *      activation: The GELU(Gaussian Error Linear Units) activation function.
 *      dense2:     The second dense layer.
 *                         output2[batch_size,out_features] = output[batch_size, hidden_features] * weight2[out_features, hidden_features] + bias2[out_features
 *   Parameters:
 *      input  (torch.Tensor): (∗,Hin ) where ∗ is batch_size and Hin=in_features
 *      weight (torch.Tensor): the learnable weights of the module of shape(out_features,in_features).
 *      bias   (torch.Tensor): the learnable bias of the module of shape(out_features)
 *   
 *   Output: (*,Hout ) where all but the last dimension are the same shape as the input and Hout = out_features.
 *
  **************************************************************************/
std::vector<at::Tensor> linear_gelu_linear_forward(at::Tensor input,   at::Tensor weight, at::Tensor bias,  
                                                   at::Tensor weight2, at::Tensor bias2)
{
  const float alpha = 1.0, beta = 0.0;

  int64_t  batch_size      = input.size(0);    // input[batch_size, in_features] 
  int64_t  in_features     = input.size(1);    // bias[hidden_features] and bias2[out_features]
  int64_t  hidden_features = weight.size(0);   // weight[hidden_features, in_features]
  int64_t  out_features    = weight2.size(0);  // weight2[out_features, hidden_features]


  at::Tensor dummy_gelu = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

  // **********************************************************************************
  // output[batch_size, hidden_features] = input[batch_size, in_features] * weight[hidden_features,in_features] + bias[hidden_features]
  // **********************************************************************************
  at::Tensor output  = at::zeros({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  at::Tensor gelu    = at::zeros({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

  // **********************************************************************************
  // output2[batch_size,out_features] = output[batch_size, hidden_features] * weight2[out_features, hidden_features] + bias2[out_features]
  // **********************************************************************************
  at::Tensor output2 = at::zeros({batch_size,out_features}, torch::device(torch::kCUDA).dtype(input.scalar_type())); // output2[batch_size,out_features]

#if DEBUG
  std::cout << "linear_gelu_linear_forward " << std::endl;
#endif
  if (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt) {
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_T, HIPBLAS_OP_N, &alpha, &beta, weight, input, output, bias, gelu, true, false, true));
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_T, HIPBLAS_OP_N, &alpha, &beta, weight2, output, output2, bias2, dummy_gelu, true, false, false));
  } else {
    std::cout << "linear_gelu_linear_forward not implimented for non-MI300 GPU" << std::endl;
  }
  return {output, output2, gelu};
}

/****************************************************************************
 * In the backward pass, we compute the gradients of the loss with respect to input, weight, and bias.
 * The key matrix operations are:
 * For second gemm
 *  1. Gradient of Input   (dX): grad_output[batch_size, hidden_features]  = output2[batch_size,out_features] ⋅ weight2[out_features, hidden_features]
 *  2. Gradient of Weights (dW): grad_weight[hidden_features, in_features] = output[batch_size, hidden_features](T) ⋅ output2[batch_size,out_features] 
 * For First gemm
 *  1. Gradient of Input   (dX): grad_input[batch_size, in_features]       = output[batch_size, hidden_features] ⋅ weight[hidden_features,in_features](T)
 *  2. Gradient of Weights (dW): grad_weight[hidden_features, in_features] = input[batch_size, in_features](T) ⋅ output[batch_size, hidden_features] 
 **************************************************************************/
std::vector<at::Tensor> linear_gelu_linear_backward(at::Tensor input, at::Tensor gelu, at::Tensor output, at::Tensor weight,
                                                    at::Tensor weight2, at::Tensor output2)
{
  const float alpha = 1.0, beta = 0.0;
  
  int64_t batch_size      = input.size(0);
  int64_t in_features     = input.size(1);
  int64_t hidden_features = weight.size(0);
  int64_t out_features    = weight2.size(0);

  hipblasStatus_t status = HIPBLAS_STATUS_NOT_INITIALIZED;

  hipblasOperation_t trans_a = HIPBLAS_OP_T;
  hipblasOperation_t trans_b = HIPBLAS_OP_N;

  at::Tensor grad_weight  = at::zeros({hidden_features, in_features},  torch::device(torch::kCUDA).dtype(input.scalar_type()));
  at::Tensor grad_weight2 = at::zeros({out_features, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  at::Tensor grad_bias    = at::zeros({hidden_features},               torch::device(torch::kCUDA).dtype(input.scalar_type()));
  at::Tensor grad_bias2   = at::zeros({out_features},                  torch::device(torch::kCUDA).dtype(input.scalar_type()));
  at::Tensor grad_input   = at::zeros({batch_size, in_features},       torch::device(torch::kCUDA).dtype(input.scalar_type()));
  at::Tensor grad_output  = at::zeros({batch_size, hidden_features},   torch::device(torch::kCUDA).dtype(input.scalar_type()));

  at::Tensor dummy_gelu = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
#if DEBUG
  std::cout << "linear_gelu_linear_backward " << std::endl;
#endif
  if (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt) {
  // **********************************************************************************
  // Gradient For second gemm  :
  // grad_output[batch_size, hidden_features]  = output2[batch_size,out_features] ⋅ weight2[out_features, hidden_features]
  // grad_weight[out_features,in_features] = input[batch_size, in_features](T)  * output[batch_size, out_features] 
  // **********************************************************************************
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_N, HIPBLAS_OP_N, &alpha, &beta, weight2, output2, grad_output, grad_bias2, dummy_gelu, false, false, false));
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_N, HIPBLAS_OP_T, &alpha, &beta, output2, output, grad_weight2, grad_bias2, dummy_gelu, true, false, false));
    grad_bias2 = output2.sum(0, false);   // ToDo: Check why HipBLASLt fail to get bgrad above so this step is not needed.

  // **********************************************************************************
  // Gradient For First gemm  :
  // grad_input [batch_size, in_features] = output[batch_size, out_features] * Weight[out_features,in_features]
  // grad_weight[out_features,in_features] = input[batch_size, in_features](T)  * output[batch_size, out_features] 
  // **********************************************************************************
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_N, HIPBLAS_OP_N, &alpha, &beta, weight, output, grad_input, grad_bias2, dummy_gelu, false, false, false));
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_N, HIPBLAS_OP_T, &alpha, &beta, output, input, grad_weight, grad_bias2, dummy_gelu, true, false, false));
    grad_bias = output.sum(0, false);   // ToDo: Check why HipBLASLt fail to get bgrad above so this step is not needed.
  } else {
    std::cout << "linear_gelu_linear_backward not implimented for non-MI300 GPU" << std::endl;
  }
  return {grad_input, grad_weight, grad_bias, grad_weight2, grad_bias2};
}
