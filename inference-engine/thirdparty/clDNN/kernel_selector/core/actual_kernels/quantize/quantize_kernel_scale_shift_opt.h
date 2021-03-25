﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "quantize_kernel_base.h"

namespace kernel_selector {

class QuantizeKernelScaleShift : public QuantizeKernelBase {
public:
    using Parent = QuantizeKernelBase;

    QuantizeKernelScaleShift() : QuantizeKernelBase("quantize_gpu_scale_shift_opt") {}
    virtual ~QuantizeKernelScaleShift() {}

    JitConstants GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const override;
    CommonDispatchData SetDefault(const quantize_params& params, const optional_params&) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
