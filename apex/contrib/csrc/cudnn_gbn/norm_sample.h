/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include <cuda_runtime.h>
#include <assert.h>
#include <tuple>
#include <functional>
#include <vector>

#include <cudnn.h>

void
run_batch_norm_forward(
    int64_t *perChannelSum,
    int64_t *epsilon,
    int64_t *tensorDims,
    int64_t *peerDims,

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
    int rank_id
);

void
run_batch_norm_backward(
    int64_t *perChannelSum,
    int64_t *epsilon,
    int64_t *tensorDims,
    int64_t *peerDims,

    void *xDevPtr,
    void *dyDevPtr,
    void *scaledevPtr,
    void *saved_meandevPtr,
    void *saved_inv_vardevPtr,
    void *dxDevPtr,
    void *dscaledevPtr,
    void *dbiasdevPtr,
    const std::vector<void*> &peer_devPtrs,
    double epsilon_val,
    int rank_id
);