// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for clDNN plugin
 *        To use in SetConfig() method of plugins
 *
 * @file cldnn_config.hpp
 */
#pragma once

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

namespace Metrics {

#define DECLARE_GPU_METRIC_KEY(name, ...) DECLARE_METRIC_KEY(GPU_##name, __VA_ARGS__)
#define GPU_METRIC_KEY(name) METRIC_KEY(GPU_##name)

/**
 * @brief Metric to get a type of GPU (integrated or discrete). The return value is either "iGPU" or "dGPU" string
 */
DECLARE_GPU_METRIC_KEY(DEVICE_TYPE, std::string);

/**
 * @brief Metric to check if DP4A instruction is supported by selected GPU device
 */
DECLARE_GPU_METRIC_KEY(SUPPORTS_DP4A, bool);

/**
 * @brief Metric to get GFX IP version in format "major.minor.revision" string
 */
DECLARE_GPU_METRIC_KEY(GFX_VERSION, std::string);

/**
 * @brief Metric to get device id. Returns string with hex ID
 */
DECLARE_GPU_METRIC_KEY(DEVICE_ID, std::string);

/**
 * @brief Metric to get global GPU memory size in bytes.
 */
DECLARE_GPU_METRIC_KEY(GLOBAL_MEM_SIZE, uint64_t);

/**
 * @brief Metric to get number of slices on current GPU.
 */
DECLARE_GPU_METRIC_KEY(NUM_SLICES, uint32_t);

/**
 * @brief Metric to get number of sub-slices per slice on current GPU.
 */
DECLARE_GPU_METRIC_KEY(NUM_SUB_SLICES_PER_SLICE, uint32_t);

/**
 * @brief Metric to get number execution units per sub-slice on current GPU.
 */
DECLARE_GPU_METRIC_KEY(NUM_EUS_PER_SUB_SLICE, uint32_t);

/**
 * @brief Metric to get number HW threads per EU on current GPU.
 */
DECLARE_GPU_METRIC_KEY(NUM_THREADS_PER_EU, uint32_t);

}  // namespace Metrics

/**
 * @brief GPU plugin configuration
 */
namespace CLDNNConfigParams {

/**
* @brief shortcut for defining configuration keys
*/
#define CLDNN_CONFIG_KEY(name) InferenceEngine::CLDNNConfigParams::_CONFIG_KEY(CLDNN_##name)
#define DECLARE_CLDNN_CONFIG_KEY(name) DECLARE_CONFIG_KEY(CLDNN_##name)
#define DECLARE_CLDNN_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(CLDNN_##name)

/**
* @brief This key instructs the clDNN plugin to use the OpenCL queue priority hint
* as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf
* this option should be used with an unsigned integer value (1 is lowest priority)
* 0 means no priority hint is set and default queue is created.
*/
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_PRIORITY);

/**
* @brief This key instructs the clDNN plugin to use throttle hints the OpenCL queue throttle hint
* as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
* chapter 9.19. This option should be used with an unsigned integer value (1 is lowest energy consumption)
* 0 means no throttle hint is set and default queue created.
*/
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_THROTTLE);

/**
* @brief This key controls clDNN memory pool optimization.
* Turned off by default.
*/
DECLARE_CLDNN_CONFIG_KEY(MEM_POOL);

/**
* @brief This key defines the directory name to which clDNN graph visualization will be dumped.
*/
DECLARE_CLDNN_CONFIG_KEY(GRAPH_DUMPS_DIR);

/**
* @brief This key defines the directory name to which full program sources will be dumped.
*/
DECLARE_CLDNN_CONFIG_KEY(SOURCES_DUMPS_DIR);

/**
* @brief This key enables FP16 precision for quantized models.
* By default the model is converted to FP32 precision before running LPT. If this key is enabled (default), then non-quantized layers
* will be converted back to FP16 after LPT, which might imrpove the performance if a model has a lot of compute operations in
* non-quantized path. This key has no effect if current device doesn't have INT8 optimization capabilities.
*/
DECLARE_CLDNN_CONFIG_KEY(ENABLE_FP16_FOR_QUANTIZED_MODELS);

/**
* @brief This key should be set to correctly handle NV12 input without pre-processing.
* Turned off by default.
*/
DECLARE_CLDNN_CONFIG_KEY(NV12_TWO_INPUTS);

/**
* @brief This key sets the max number of host threads that can be used by GPU plugin on model loading.
* Default value is maximum number of threads available in the environment.
*/
DECLARE_CLDNN_CONFIG_KEY(MAX_NUM_THREADS);

}  // namespace CLDNNConfigParams
}  // namespace InferenceEngine
