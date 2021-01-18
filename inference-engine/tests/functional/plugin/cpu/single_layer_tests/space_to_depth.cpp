// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/space_to_depth.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::spaceToDepthParamsTuple,
        CPUSpecificParams
> SpaceToDepthLayerCPUTestParamSet;

class SpaceToDepthLayerCPUTest : public testing::WithParamInterface<SpaceToDepthLayerCPUTestParamSet>,
                        virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SpaceToDepthLayerCPUTestParamSet> obj) {
        LayerTestsDefinitions::spaceToDepthParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::SpaceToDepthLayerTest::getTestCaseName(
                testing::TestParamInfo<LayerTestsDefinitions::spaceToDepthParamsTuple>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        LayerTestsDefinitions::spaceToDepthParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<size_t> inputShape;
        SpaceToDepth::SpaceToDepthMode mode;
        std::size_t blockSize;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, netPrecision, mode, blockSize, targetDevice) = basicParamsSet;

        inPrc = outPrc = netPrecision;
        selectedType = std::string("unknown_") + netPrecision.name();
        auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(inPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto d2s = ngraph::builder::makeSpaceToDepth(paramOuts[0], mode, blockSize);
        d2s->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(d2s)};
        function = std::make_shared<ngraph::Function>(results, params, "SpaceToDepth");
    }
};

TEST_P(SpaceToDepthLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "SpaceToDepth");
}

namespace {

const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {nChw8c}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {ndhwc}, {}, {}};


const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

const std::vector<SpaceToDepth::SpaceToDepthMode> spaceToDepthModes = {
        SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST
};

const std::vector<std::vector<size_t>> inputShapesBS2_4D = {
        {1, 16, 2, 2}, {1, 16, 4, 2}, {1, 32, 6, 8}, {2, 32, 4, 6}, {2, 64, 8, 2},
};

const std::vector<std::vector<size_t >> inputShapesBS3_4D = {
        {1, 1, 3, 3}, {1, 3, 3, 6}, {1, 5, 6, 3}, {2, 5, 9, 3}, {3, 5, 6, 6}
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc,
};

const auto spaceToDepthBS2_4DParams = testing::Combine(
        testing::ValuesIn(inputShapesBS2_4D),
        testing::ValuesIn(inputPrecisions),
        testing::ValuesIn(spaceToDepthModes),
        testing::Values(1, 2),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_CPUSpaceToDepthBS2_4D,
        SpaceToDepthLayerCPUTest,
        ::testing::Combine(
                spaceToDepthBS2_4DParams,
                ::testing::ValuesIn(CPUParams4D)),
        SpaceToDepthLayerCPUTest::getTestCaseName
);

const auto spaceToDepthBS3_4DParams = testing::Combine(
        testing::ValuesIn(inputShapesBS3_4D),
        testing::ValuesIn(inputPrecisions),
        testing::ValuesIn(spaceToDepthModes),
        testing::Values(1, 3),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_CPUSpaceToDepthBS3_4D,
        SpaceToDepthLayerCPUTest,
        ::testing::Combine(
                spaceToDepthBS3_4DParams,
                ::testing::Values(cpuParams_nhwc)),
        SpaceToDepthLayerCPUTest::getTestCaseName
);

const std::vector<std::vector<size_t >> inputShapesBS2_5D = {
        {1, 16, 2, 2, 2}, {1, 16, 4, 4, 2}, {1, 32, 2, 6, 2}, {2, 32, 4, 2, 2}, {2, 64, 2, 2, 6}
};

const std::vector<std::vector<size_t >> inputShapesBS3_5D = {
        {1, 1, 3, 3, 3}, {1, 2, 3, 6, 9}, {1, 5, 6, 3, 3}, {2, 5, 3, 9, 3}, {3, 5, 3, 3, 6}
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ndhwc,
};

const auto spaceToDepthBS2_5DParams = testing::Combine(
        testing::ValuesIn(inputShapesBS2_5D),
        testing::ValuesIn(inputPrecisions),
        testing::ValuesIn(spaceToDepthModes),
        testing::Values(1, 2),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_CPUSpaceToDepthBS2_5D,
        SpaceToDepthLayerCPUTest,
        ::testing::Combine(
                spaceToDepthBS2_5DParams,
                ::testing::ValuesIn(CPUParams5D)),
        SpaceToDepthLayerCPUTest::getTestCaseName
);

const auto spaceToDepthBS3_5DParams = testing::Combine(
        testing::ValuesIn(inputShapesBS3_5D),
        testing::ValuesIn(inputPrecisions),
        testing::ValuesIn(spaceToDepthModes),
        testing::Values(1, 3),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_CPUSpaceToDepthBS3_5D,
        SpaceToDepthLayerCPUTest,
        ::testing::Combine(
                spaceToDepthBS3_5DParams,
                ::testing::Values(cpuParams_ndhwc)),
        SpaceToDepthLayerCPUTest::getTestCaseName
);

} // namespace
} // namespace CPULayerTestsDefinitions
