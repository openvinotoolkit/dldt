// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#include <vector>
#include <tuple>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

using LayersRestrictionsParamsTuple = typename std::tuple<
        InferenceEngine::Precision,         // Network Precision
        std::map<std::string, std::string>, // Configuration
        std::string>;                       // Device name

namespace LayerTestsDefinitions {

struct SplitAxis {
    static const char* getName() { return "SplitAxis"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {{1, 100}});
        auto variadicSplit = ngraph::builder::makeVariadicSplit(params[0], {1}, 0);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(variadicSplit->output(0))};
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "and axis(0) not supported"; }
};

struct FullyConnectedBatchSize {
    static const char* getName() { return "FullyConnectedBatchSize"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {{85, 79}});
        auto fullyConnected = ngraph::builder::makeFullyConnected(params[0], ngPrc, 1);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fullyConnected) };
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "and batch size(85) not supported"; }
};

template<typename T>
class LayersRestrictions : public testing::WithParamInterface<LayersRestrictionsParamsTuple>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayersRestrictionsParamsTuple> obj) {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> configuration;
        std::string targetDevice;
        std::tie(netPrecision, configuration, targetDevice) = obj.param;
        std::ostringstream result;
        result << T::getName() << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "configItem=" << configItem.first << "_" << configItem.second << "_";
        }
        return result.str();
    }
    static const char* getMatch() { return T::getMatch(); }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, configuration, targetDevice) = this->GetParam();
        function = T::createTopology(netPrecision);
    }
};

using LayersRestrictionsSplitAxis = LayersRestrictions<SplitAxis>;
using LayersRestrictionsFullyConnectedBatchSize = LayersRestrictions<FullyConnectedBatchSize>;

TEST_P(LayersRestrictionsSplitAxis, CompareWithRefImpl) {
    std::string what;
    try {
        LoadNetwork();
    } catch (const std::exception& e) {
        what.assign(e.what());
    }
    EXPECT_TRUE(what.find(getMatch()) != std::string::npos);
}

TEST_P(LayersRestrictionsFullyConnectedBatchSize, CompareWithRefImpl) {
    std::string what;
    try {
        LoadNetwork();
    } catch (const std::exception& e) {
        what.assign(e.what());
    }
    EXPECT_TRUE(what.find(getMatch()) != std::string::npos);
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};
const std::vector<std::map<std::string, std::string>> configs = {
    { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"} }
};

INSTANTIATE_TEST_CASE_P(smoke_layers_restrictions, LayersRestrictionsSplitAxis,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(configs),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        LayersRestrictionsSplitAxis::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_layers_restrictions, LayersRestrictionsFullyConnectedBatchSize,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(configs),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        LayersRestrictionsFullyConnectedBatchSize::getTestCaseName);
} // namespace LayerTestsDefinitions
