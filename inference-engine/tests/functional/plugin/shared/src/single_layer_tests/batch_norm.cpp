// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/batch_norm.hpp"


namespace LayerTestsDefinitions {
std::string BatchNormLayerTest::getTestCaseName(const testing::TestParamInfo<BatchNormLayerTestParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    double epsilon;
    std::string targetDevice;
    std::tie(epsilon, netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "epsilon=" << epsilon << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr BatchNormLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), 3, 0, 1);
}

void BatchNormLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    double epsilon;
    std::tie(epsilon, netPrecision, inputShapes, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::opset4::Parameter>(params));

    auto batchNorm = ngraph::builder::makeBatchNormInference(paramOuts[0], epsilon);
    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(batchNorm)};
    function = std::make_shared<ngraph::Function>(results, params, "BatchNormInference");
}

TEST_P(BatchNormLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
