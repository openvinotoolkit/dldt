﻿// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include "binary_convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class BinaryConvolutionKernel1x1 : public BinaryConvolutionKernelBase {
public:
    using Parent = BinaryConvolutionKernelBase;

    BinaryConvolutionKernel1x1() : BinaryConvolutionKernelBase("binary_convolution_gpu_1x1") {}
    virtual ~BinaryConvolutionKernel1x1() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<WeightsLayout> GetSupportedWeightLayouts(const binary_convolution_params&) const override {
        return {WeightsLayout::os_is_yx_osv32_isv32p};
    }
    JitConstants GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                const DispatchData& kd) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const binary_convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const binary_convolution_params& params, const DispatchData& kd) const override;
};
}  // namespace kernel_selector
