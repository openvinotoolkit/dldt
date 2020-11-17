// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <string>

#include <ie_core.hpp>

#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"


typedef std::tuple<
        std::vector<std::vector<size_t>>,   // input shapes
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>  //Configuration
> EltwiseSplitOverChannelsPassParams;

namespace LayerTestsDefinitions {

class EltwiseSplitOverChannelsPassTest : public testing::WithParamInterface<EltwiseSplitOverChannelsPassParams>,
                                             public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseSplitOverChannelsPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<std::vector<size_t >> input_shapes;
        std::tie(input_shapes, netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "Shapes=" << CommonTestUtils::vec2str(input_shapes) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<std::vector<size_t >> input_shapes;
        std::tie(input_shapes, netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, { {input_shapes[0]} });
        auto const_mult2 = ngraph::builder::makeConstant<float>(ngPrc, input_shapes[1], {-1.0f});

        auto sum = ngraph::builder::makeEltwise(params[0], const_mult2, ngraph::helpers::EltwiseTypes::MULTIPLY);
        function = std::make_shared<ngraph::Function>(sum, params, "RemovePermutationPass");
    }
};

TEST_P(EltwiseSplitOverChannelsPassTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<std::vector<size_t>>> shapes {
        {{1, 67000}, {1, 67000}},
        {{1, 70000}, {1, 70000}},
        {{1, 67000}, {}},
        {{1, 70000}, {}}
};

const std::vector<std::map<std::string, std::string>> configs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_COMPACT_MODE", "NO"},
                {"GNA_SCALE_FACTOR_0", "2048"}
        }
};

INSTANTIATE_TEST_CASE_P(smoke_EltwiseSplitOverChennels, EltwiseSplitOverChannelsPassTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(configs)),
                        EltwiseSplitOverChannelsPassTest::getTestCaseName);

} // namespace LayerTestsDefinitions

