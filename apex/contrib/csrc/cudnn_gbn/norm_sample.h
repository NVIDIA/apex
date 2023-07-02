#pragma once

/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <tuple>
#include <functional>

#include <cudnn.h>
#include <cudnn_frontend.h>

/* some helpers
 */
void generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat);

int64_t checkCudaError(cudaError_t code, const char* expr, const char* file, int line);
int64_t checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line);

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int64_t err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        assert(err == 0);                                                       \
    } while (0)

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int64_t err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        assert(err == 0);                                                        \
    } while (0)

/**
 * @brief Run a Group BN forward sample with 2 peer stat tensors.
 *
 * @param tensorDims an array with shape (N, C, H, W) for input tensor dims. Stride in NHWC or NCHW will take care of memory format
 * @param perChannelSum an array with shape (1, C, 1, 1) to denote the sum values for each channel in the input tensor
 * @param epsilon a scalar array with shape (1, 1, 1, 1) to represent the epsilon value for the BN
 * @param peerDims an array with shape (num GPUs, 2 * C, 1, 1) to denote the tensor dimensions for peer stat tensor in GBN

 *
 */
cudnn_frontend::ExecutionPlan run_batch_norm_forward(
						     int64_t *tensorDims,
						     int64_t *perChannelSum,
						     int64_t *epsilon,
						     int64_t *peerDims,
						     cudnnDataType_t in_out_data_type);
/**
 * @param xDevPtr input tensor device pointer
 * @param yDevPtr output tensor device pointer
 * @param scaledevPtr input scale device pointer for BN scaling
 * @param biasdevPtr input scale device pointer for BN bias
 * @param in_meandevPtr Input mean device pointer
 * @param in_vardevPtr Input variance device pointer
 * @param out_meandevPtr output mean device pointer
 * @param out_vardevPtr output variance device pointer
 * @param saved_meandevPtr saved mean device pointer for BN backward
 * @param saved_inv_vardevPtr saved inverse variance device pointer for BN backward
 * @param peer_devPtr1 peer stat tensor 1 device pointer
 * @param peer_devPtr2 peer stat tensor 2 device pointer
 * @param epsilon_val episilon value as a double
 * @param exponential_decay_factor exponential_decay_factor as a value
 *
**/
void execute_batch_norm_forward(cudnn_frontend::ExecutionPlan plan,
				void *xDevPtr,
				void *yDevPtr,
				void *scaledevPtr,
				void *biasdevPtr,
				void *in_meandevPtr,
				void *in_vardevPtr,
				void *out_meandevPtr,
				void *out_vardevPtr,
				void *saved_meandevPtr,
				void *saved_inv_vardevPtr,
				const std::vector<void*> &peer_devPtrs,
				double epsilon_val,
				double exponential_decay_factor,
				size_t peer_size,
				int rank_id);

/**
 * @brief Run a Group BN backward sample with 2 peer stat tensors.
 *
 * @param tensorDims an array with shape (N, C, H, W) for input tensor dims. Stride in NHWC or NCHW will take care of memory format
 * @param perChannelSum an array with shape (1, C, 1, 1) to denote the sum values for each channel in the input tensor
 * @param epsilon a scalar array with shape (1, 1, 1, 1) to represent the epsilon value for the BN
 * @param peerDims an array with shape (num GPUs, 2 * C, 1, 1) to denote the tensor dimensions for peer stat tensor in GBN
    *
*/
cudnn_frontend::ExecutionPlan run_batch_norm_backward(int64_t *tensorDims,
						      int64_t *perChannelSum,
						      int64_t *epsilon,
						      int64_t *peerDims,
						      cudnnDataType_t data_type);

/**
 * @brief Run a Group BN backward sample with 2 peer stat tensors.
 *
 * @param xDevPtr input tensor device pointer
 * @param yDevPtr output tensor device pointer
 * @param scaledevPtr input scale device pointer for BN scaling
 * @param biasdevPtr input scale device pointer for BN bias
 * @param in_meandevPtr Input mean device pointer
 * @param in_vardevPtr Input variance device pointer
 * @param out_meandevPtr output mean device pointer
 * @param out_vardevPtr output variance device pointer
 * @param saved_meandevPtr saved mean device pointer for BN backward
 * @param saved_inv_vardevPtr saved inverse variance device pointer for BN backward
 * @param peer_devPtr1 peer stat tensor 1 device pointer
 * @param peer_devPtr2 peer stat tensor 2 device pointer
 * @param epsilon_val episilon value as a double
 *
 */
void execute_batch_norm_backward(cudnn_frontend::ExecutionPlan plan,
				 void *xDevPtr,
				 void *dyDevPtr,
				 void *scaledevPtr,
				 void *saved_meandevPtr,
				 void *saved_inv_vardevPtr,
				 const std::vector<void*> &peer_devPtrs,
				 void *dxDevPtr,
				 void *dscaledevPtr,
				 void *dbiasdevPtr,
				 double epsilon_val,
				 size_t peer_size,
				 int rank_id);
