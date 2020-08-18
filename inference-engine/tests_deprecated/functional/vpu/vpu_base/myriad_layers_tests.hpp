// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <ie_version.hpp>
#include <algorithm>
#include <cstddef>
#include <precision_utils.h>
#include <tuple>
#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>
#include "myriad_layers_reference_functions.hpp"
#include "vpu_layers_tests.hpp"
#include <file_utils.h>

/* Function to calculate CHW dimensions for the blob generated by */
/* Myriad plugin.                                            */

class myriadLayersTests_nightly : public vpuLayersTests {
protected:
    void makeSingleLayerNetwork(const LayerParams& layerParams,
                                const NetworkParams& networkParams = {},
                                const WeightsBlob::Ptr& weights = nullptr);
};

template<class T>
class myriadLayerTestBaseWithParam: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<T> {
};

/* common classes for different basic tests */
extern const char POOLING_MAX[];
extern const char POOLING_AVG[];

struct pooling_layer_params {
    param_size kernel;
    param_size stride;
    param_size pad;
};

struct nd_tensor_test_params {
    static constexpr int MAX_DIMS = 8;

    size_t dims[MAX_DIMS];
};

template <const char* poolType, typename... Types>
class PoolingTest : public myriadLayersTests_nightly,
                    public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, 
                                                       pooling_layer_params, vpu::LayoutPreference, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, vpu::LayoutPreference, Types...>>::GetParam();
        _input_tensor       = std::get<0>(p);
        _kernel_val         = std::get<1>(p).kernel;
        _stride_val         = std::get<1>(p).stride;
        _pad_val            = std::get<1>(p).pad;
        _layout_preference  = std::get<2>(p);

        if (_pad_val.x >= _kernel_val.x) {
            _pad_val.x = _kernel_val.x - 1;
        }
        if (_pad_val.y >= _kernel_val.y) {
            _pad_val.y = _kernel_val.y - 1;
        }

        _params["kernel-x"] = std::to_string(_kernel_val.x);
        _params["kernel-y"] = std::to_string(_kernel_val.y);
        _params["stride-x"] = std::to_string(_stride_val.x);
        _params["stride-y"] = std::to_string(_stride_val.y);
        _params["pad-x"] = std::to_string(_pad_val.x);
        _params["pad-y"] = std::to_string(_pad_val.y);
        _params["kernel"]   = std::to_string(_kernel_val.y) + "," + std::to_string(_kernel_val.x);
        _params["strides"]  = std::to_string(_stride_val.y) + "," + std::to_string(_stride_val.x);
        _params["pads_begin"] = std::to_string(_pad_val.y) + "," + std::to_string(_pad_val.x);
        _params["pads_end"] = std::to_string(_pad_val.y) + "," + std::to_string(_pad_val.x);
        _params["pool-method"] = poolType;
        const bool isMaxPool = poolType == std::string("max");
        if (!isMaxPool)
            _params["exclude-pad"] = "true";
        _output_tensor.resize(4);
        _output_tensor[3] = std::ceil((_input_tensor[3] + 2. * _pad_val.x - _kernel_val.x) / _stride_val.x + 1);
        _output_tensor[2] = std::ceil((_input_tensor[2] + 2. * _pad_val.y - _kernel_val.y) / _stride_val.y + 1);
        _output_tensor[1] = _input_tensor[1];
        _output_tensor[0] = 1;
        ASSERT_EQ(_input_tensor.size(), 4);
        _testNet.addLayer(LayerInitParams(_irVersion == IRVersion::v10 ? (isMaxPool ? "MaxPool" : "AvgPool") : "Pooling")
                 .params(_params)
                 .in({_input_tensor})
                 .out({_output_tensor}),
                 ref_pooling_wrap);
    }

    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    param_size _kernel_val;
    param_size _stride_val;
    param_size _pad_val;
    vpu::LayoutPreference _layout_preference;
    std::map<std::string, std::string> _params;
};

using KernelSizeParam = param_size;
using PadsParam = param_size;
using StridesParam = param_size;
using GlobalPoolingTestParam = std::tuple<InferenceEngine::SizeVector, KernelSizeParam, PadsParam, StridesParam>;

template <const char* poolType/*, typename... Types*/>
class GlobalPoolingTest : public myriadLayersTests_nightly,
                    public testing::WithParamInterface<GlobalPoolingTestParam>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
         auto params = ::testing::WithParamInterface<GlobalPoolingTestParam>::GetParam();
        _input_tensor = std::get<0>(params);
        _kernel_val   = std::get<1>(params);
        _pad_val      = std::get<2>(params);
        _stride_val   = std::get<3>(params);

#if 0 // 4DGP
        // TODO: make it the test argument
        _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NCHW);
//        _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NHWC);
#endif

        _params["kernel-x"] = std::to_string(_kernel_val.x);
        _params["kernel-y"] = std::to_string(_kernel_val.y);
        _params["stride-x"] = std::to_string(_stride_val.x);
        _params["stride-y"] = std::to_string(_stride_val.y);
        _params["pad-x"] = std::to_string(_pad_val.x);
        _params["pad-y"] = std::to_string(_pad_val.y);
        _params["pool-method"] = poolType;
        _output_tensor.resize(4);
        _output_tensor[3] = std::floor((_input_tensor[3] + 2. * _pad_val.x - _kernel_val.x) / _stride_val.x) + 1;
        _output_tensor[2] = std::floor((_input_tensor[2] + 2. * _pad_val.y - _kernel_val.y) / _stride_val.y) + 1;
        _output_tensor[1] = _input_tensor[1];
        _output_tensor[0] = _input_tensor[0];
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
        ASSERT_EQ(_input_tensor.size(), 4);
        _testNet.addLayer(LayerInitParams("Pooling")
                 .params(_params)
                 .in({_input_tensor})
                 .out({_output_tensor}),
                 ref_pooling_wrap);
    }

    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    param_size _kernel_val;
    param_size _stride_val;
    param_size _pad_val;
    std::map<std::string, std::string> _params;
};

template <const char* poolType, const bool excludePad = false, typename... Types>
class PoolingTestPad4 : public myriadLayersTests_nightly,
                    public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, paddings4, vpu::LayoutPreference, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, paddings4, vpu::LayoutPreference, Types...>>::GetParam();
        _input_tensor       = std::get<0>(p);
        _kernel_val         = std::get<1>(p);
        _stride_val         = std::get<2>(p);
        _pad_val            = std::get<3>(p);
        _layout_preference  = std::get<4>(p);

        if (_pad_val.left >= _kernel_val.x) {
            _pad_val.left = _kernel_val.x - 1;
        }
        if (_pad_val.right >= _kernel_val.x) {
            _pad_val.right = _kernel_val.x - 1;
        }
        if (_pad_val.top >= _kernel_val.y) {
            _pad_val.top = _kernel_val.y - 1;
        }
        if (_pad_val.bottom >= _kernel_val.y) {
            _pad_val.bottom = _kernel_val.y - 1;
        }
        auto bool2str = [](bool value) {
            return value ? "true" : "false";
        };
        _params["kernel-x"] = std::to_string(_kernel_val.x);
        _params["kernel-y"] = std::to_string(_kernel_val.y);
        _params["stride-x"] = std::to_string(_stride_val.x);
        _params["stride-y"] = std::to_string(_stride_val.y);
        _params["pad-x"] = std::to_string(_pad_val.left);
        _params["pad-y"] = std::to_string(_pad_val.top);
        _params["exclude-pad"] = bool2str(excludePad);
        _params["pool-method"] = poolType;
        _output_tensor.resize(4);
        _output_tensor[3] = std::ceil((_input_tensor[3] + _pad_val.left + _pad_val.right  - _kernel_val.x) / _stride_val.x + 1);
        _output_tensor[2] = std::ceil((_input_tensor[2] + _pad_val.top  + _pad_val.bottom - _kernel_val.y) / _stride_val.y + 1);
        _output_tensor[1] = _input_tensor[1];
        _output_tensor[0] = 1;
        ASSERT_EQ(_input_tensor.size(), 4);
        _testNet.addLayer(LayerInitParams("Pooling")
                 .params(_params)
                 .in({_input_tensor})
                 .out({_output_tensor}),
                 ref_pooling_wrap);
    }

    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    param_size _kernel_val;
    param_size _stride_val;
    paddings4 _pad_val;
    vpu::LayoutPreference _layout_preference;
    std::map<std::string, std::string> _params;
};

template <typename... Types>
class ConvolutionTest : public myriadLayersTests_nightly,
                        public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, Types...>>::GetParam();
        _input_tensor = std::get<0>(p);
        kernel = std::get<1>(p);
        param_size stride = std::get<2>(p);
        param_size pad = std::get<3>(p);
        size_t out_channels = std::get<4>(p);
        group = std::get<5>(p);
        get_dims(_input_tensor, IW, IH,IC);
        size_t out_w = (IW + 2 * pad.x - kernel.x + stride.x) / stride.x;
        size_t out_h = (IH + 2 * pad.y - kernel.y + stride.y) / stride.y;

        gen_dims(_output_tensor, _input_tensor.size(), out_w, out_h, out_channels);

        size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
        size_t num_bias    = out_channels;

        std::map<std::string, std::string> layer_params = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"dilations", "1,1"}
                , {"strides", std::to_string(stride.y) + "," + std::to_string(stride.x)}
                , {"pads_begin", std::to_string(pad.y) + "," + std::to_string(pad.x)}
                , {"pads_end", std::to_string(pad.y) + "," + std::to_string(pad.x)}
                , {"output", std::to_string(out_channels)}
                , {"group", std::to_string(group)}
        };
        _testNet.addLayer(LayerInitParams("Convolution")
                 .params(layer_params)
                 .in({_input_tensor})
                 .out({_output_tensor})
                 .weights(num_weights).fillWeights(defaultWeightsRange)
                 .biases(num_bias).fillBiases(defaultWeightsRange)
                 .weightsDim({{out_channels, (IC / group), kernel.y, kernel.x}})
                 .biasesDim({{1, out_channels, 1, 1}}),
                 ref_convolution_wrap);
    }
    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    size_t  group = 0;
    param_size kernel;
};

template <typename... Types>
class FCTest : public myriadLayersTests_nightly,
               public testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, Types...>>::GetParam();
        _par = std::get<0>(p);
        int32_t input_dim = std::get<1>(p);
        int32_t add_bias = std::get<2>(p);
        std::map<std::string, std::string> params;
        params["out-size"] = std::to_string(_par.out_c);
        int32_t IW = _par.in.w;
        int32_t IH = _par.in.h;
        int32_t IC = _par.in.c;
        gen_dims(_input_tensor, input_dim, IW, IH, IC);

        _output_tensor.push_back(1);
        _output_tensor.push_back(_par.out_c);

        size_t sz_weights = IC * IH * IW * _par.out_c;
        size_t sz_bias = 0;
        if (add_bias) {
            sz_bias = _par.out_c;
        }
        size_t sz = sz_weights + sz_bias;
        // @todo: FullyConnected is not present in IRv10. Need to move to MatMul somehow. MatMul need different initializetion here.
        _testNet.addLayer(LayerInitParams(_irVersion == IRVersion::v10 ? "MatMul" : "FullyConnected")
                  .params(params)
                  .in({_input_tensor})
                  .out({_output_tensor})
                 .weights(sz_weights).fillWeights(defaultWeightsRange).weightsDim({{1U, 0U + IC * _par.out_c, 0U + IH, 0U + IW}})
                 .biases(sz_bias).fillBiases(defaultWeightsRange).biasesDim({{sz_bias}}),
                 ref_innerproduct_wrap);
    }
    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    fcon_test_params _par;
};

/* parameters definitions for the tests with several layers within the NET */
extern const std::vector<InferenceEngine::SizeVector> g_poolingInput;
extern const std::vector<InferenceEngine::SizeVector> g_poolingInput_postOp;
extern const std::vector<pooling_layer_params> g_poolingLayerParamsFull;
extern const std::vector<pooling_layer_params> g_poolingLayerParamsLite;
extern const std::vector<vpu::LayoutPreference> g_poolingLayout;
extern const std::vector<InferenceEngine::SizeVector> g_convolutionTensors;
extern const std::vector<InferenceEngine::SizeVector> g_convolutionTensors_postOp;
extern const std::vector<fcon_test_params> g_fcTestParamsSubset;
extern const std::vector<int32_t> g_dimensionsFC;
extern const std::vector<int32_t> g_addBiasFC;
