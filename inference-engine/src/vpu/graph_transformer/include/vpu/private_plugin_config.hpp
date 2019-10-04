// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <vpu/vpu_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUConfigParams {

//
// Common options
//

// Compilation

DECLARE_VPU_CONFIG_KEY(NUMBER_OF_SHAVES);
DECLARE_VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES);
DECLARE_VPU_CONFIG_KEY(TENSOR_STRIDES);

DECLARE_VPU_CONFIG_KEY(HW_ADAPTIVE_MODE);

DECLARE_VPU_CONFIG_KEY(PERF_REPORT_MODE);
DECLARE_VPU_CONFIG_VALUE(PER_LAYER);
DECLARE_VPU_CONFIG_VALUE(PER_STAGE);

// Optimizations

DECLARE_VPU_CONFIG_KEY(COPY_OPTIMIZATION);
DECLARE_VPU_CONFIG_KEY(HW_INJECT_STAGES);
DECLARE_VPU_CONFIG_KEY(HW_POOL_CONV_MERGE);
DECLARE_VPU_CONFIG_KEY(PACK_DATA_IN_CMX);

DECLARE_VPU_CONFIG_KEY(HW_DILATION);

// Debug

DECLARE_VPU_CONFIG_KEY(DETECT_NETWORK_BATCH);

DECLARE_VPU_CONFIG_KEY(HW_WHITE_LIST);
DECLARE_VPU_CONFIG_KEY(HW_BLACK_LIST);

DECLARE_VPU_CONFIG_KEY(NONE_LAYERS);

DECLARE_VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS);

//
// Myriad plugin options
//

// Power Manager

DECLARE_VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT);

DECLARE_VPU_MYRIAD_CONFIG_VALUE(POWER_FULL);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(POWER_INFER);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE_SHAVES);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE_NCES);

DECLARE_VPU_MYRIAD_CONFIG_KEY(WATCHDOG);
INFERENCE_ENGINE_DEPRECATED
DECLARE_VPU_CONFIG_KEY(WATCHDOG);

DECLARE_VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
