﻿// Copyright (c) 2018 Intel Corporation
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


#include "mvn_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"

#include <algorithm>
#include <vector>
#include <string>

namespace kernel_selector {
ParamsKey MVNKernelBfyxOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNMode(MVNMode::ACROSS_CHANNELS);
    k.EnableMVNNormalizeVariance();
    return k;
}

MVNKernelBfyxOpt::Parent::DispatchData MVNKernelBfyxOpt::SetDefault(const mvn_params& params) const {
    DispatchData kd;

    const auto& input = params.inputs[0];

    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        kd.dataSetSize = input.X().v * input.Y().v * input.Z().v;
        kd.dataSetsCount = input.Batch().v * input.Feature().v;
    } else {
        kd.dataSetSize = input.X().v * input.Y().v * input.Z().v * input.Feature().v;
        kd.dataSetsCount = input.Batch().v;
    }

    // start with 1 thread per data set
    kd.gws0 = 1;
    kd.gws1 = kd.dataSetsCount;
    kd.gws2 = 1;
    kd.itemsNum = kd.dataSetSize;

    // We have two units of data per work item in current implementation.
    auto local_mem_per_wi = 2 * BytesPerElement(params.inputs[0].GetDType());
    // Combining device execution and local memory restrictions to compute maximum possible LWS.
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = 1;
    // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory
    // reads.
    while ((kd.itemsNum > 32 || kd.lws0 < kd.itemsNum) && (2 * kd.lws0 <= max_lws)) {
        kd.lws0 *= 2;
        kd.itemsNum /= 2;
    }

    kd.gws0 = kd.lws0;
    kd.leftovers = kd.dataSetSize % kd.lws0;

    return kd;
}

JitConstants MVNKernelBfyxOpt::GetJitConstants(const mvn_params& params, MVNKernelBase::DispatchData kd) const {
    auto jit = MVNKernelBase::GetJitConstants(params, kd);

    jit.AddConstants({
        MakeJitConstant("ITEMS_NUM", kd.itemsNum),
        MakeJitConstant("LWS", kd.lws0),
        MakeJitConstant("GWS", kd.gws0),
        MakeJitConstant("DATA_SETS_COUNT", kd.dataSetsCount),
        MakeJitConstant("DATA_SET_SIZE", kd.dataSetSize),
        MakeJitConstant("LEFTOVERS", kd.leftovers),
    });
    auto activation_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.inputs[0].GetDims().size() <= 4) {
            if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
                idx_order = { "(data_set_idx / OUTPUT_FEATURE_NUM)",
                              "(data_set_idx % OUTPUT_FEATURE_NUM)",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            } else if (params.mvnMode == MVNMode::ACROSS_CHANNELS) {
                idx_order = { "data_set_idx",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y))",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            }
        } else if (params.inputs[0].GetDims().size() == 5) {
            if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
                idx_order = { "(data_set_idx / OUTPUT_FEATURE_NUM)",
                              "(data_set_idx % OUTPUT_FEATURE_NUM)",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y))",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            } else if (params.mvnMode == MVNMode::ACROSS_CHANNELS) {
                idx_order = { "data_set_idx",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z))",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z)",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            }
        }
        auto conf = FusedOpsConfiguration("", idx_order, "result", activation_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::DISABLED);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData MVNKernelBfyxOpt::GetKernelsData(const Params& params, const optional_params& optParams) const {
    return GetCommonKernelsData(params, optParams, FORCE_PRIORITY_7);
}
}  // namespace kernel_selector
