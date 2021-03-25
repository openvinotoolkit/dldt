﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_selector.h"
#include "reorder_kernel.h"
#include "reorder_kernel_fast_b1.h"
#include "reorder_from_winograd_2x3_kernel.h"
#include "reorder_to_winograd_2x3_kernel.h"
#include "reorder_kernel_to_yxfb_batched.h"
#include "reorder_kernel_binary.h"
#include "reorder_biplanar_nv12.h"
#include "reorder_kernel_fs_b_yx_fsv32_to_bfyx.h"

namespace kernel_selector {

reorder_kernel_selector::reorder_kernel_selector() {
    Attach<ReorderKernelRef>();
    Attach<ReorderKernelBinary>();
    Attach<ReorderKernelFastBatch1>();
    Attach<ReorderFromWinograd2x3Kernel>();
    Attach<ReorderToWinograd2x3Kernel>();
    Attach<ReorderKernel_to_yxfb_batched>();
    Attach<reorder_biplanar_nv12>();
    Attach<ReorderKernel_fs_b_yx_fsv32_to_bfyx>();
}

KernelsData reorder_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REORDER);
}
}  // namespace kernel_selector
