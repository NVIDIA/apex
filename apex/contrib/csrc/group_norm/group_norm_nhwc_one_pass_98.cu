/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "group_norm_nhwc_bwd_one_pass_kernel.cuh"
#include "group_norm_nhwc_fwd_one_pass_kernel.cuh"
#include "macros.h"

GN_FWD_BWD_ONE_PASS_DEFINITION(/* CHANNELS_PER_GROUP */ 98, /* THREADS_PER_BLOCK */ 392)
