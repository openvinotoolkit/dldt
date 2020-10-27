// Copyright (c) 2020 Intel Corporation
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

#include "pooling_kernel_base.h"

namespace kernel_selector {
class PoolingKerneGPU_fs_b_yx_fsv32 : public PoolingKernelBase {
public:
    PoolingKerneGPU_fs_b_yx_fsv32() : PoolingKernelBase("pooling_gpu_fs_b_yx_fsv32") {}
    virtual ~PoolingKerneGPU_fs_b_yx_fsv32() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const pooling_params& params) const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
