// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "single_layer_tests/lrn.hpp"

namespace LayerTestsDefinitions {

std::string LrnLayerTest::getTestCaseName(testing::TestParamInfo<lrnLayerTestParamsSet> obj) {
    double alpha, beta, bias;
    size_t size;
    std::vector<size_t> axes;
    InferenceEngine::Precision  netPrecision;
    std::vector<size_t> inputShapes;
    std::string targetDevice;
    std::tie(alpha, beta, bias, size, axes, netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "Alpha=" << alpha << separator;
    result << "Beta=" << beta << separator;
    result << "Bias=" << bias << separator;
    result << "Size=" << size << separator;
    result << "Axes=" << CommonTestUtils::vec2str(axes) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void LrnLayerTest::SetUp() {
    std::vector<size_t> inputShapes;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    double alpha, beta, bias;
    size_t size;
    std::vector<size_t> axes;
    std::tie(alpha, beta, bias, size, axes, netPrecision, inputShapes, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto axes_node = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{axes.size()}, axes.data());
    auto lrn = std::make_shared<ngraph::opset3::LRN>(paramIn[0], axes_node, alpha, beta, bias, size);
    ngraph::ResultVector results {std::make_shared<ngraph::opset3::Result>(lrn)};
    function = std::make_shared<ngraph::Function>(results, params, "lrn");
}

TEST_P(LrnLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
