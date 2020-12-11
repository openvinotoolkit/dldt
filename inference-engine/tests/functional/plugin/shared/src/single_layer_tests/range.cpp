// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_precision.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "single_layer_tests/range.hpp"

namespace LayerTestsDefinitions {

std::string RangeLayerTest::getTestCaseName(testing::TestParamInfo<RangeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    float start, stop, step;
    std::string targetDevice;
    std::tie(start, stop, step, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inPRC=" << inPrc.name() << separator;
    result << "outPRC=" << outPrc.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void RangeLayerTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    auto blobStart = inferRequest.GetBlob("start");
    blobStart = FuncTestUtils::createAndFillBlobWithFloatArray(blobStart->getTensorDesc(), &start, 1);

    auto blobStop = inferRequest.GetBlob("stop");
    blobStop = FuncTestUtils::createAndFillBlobWithFloatArray(blobStop->getTensorDesc(), &stop, 1);

    auto blobStep = inferRequest.GetBlob("step");
    blobStep = FuncTestUtils::createAndFillBlobWithFloatArray(blobStep->getTensorDesc(), &step, 1);

    inferRequest.Infer();
}

void RangeLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(start, stop, step, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {std::vector<size_t>(), std::vector<size_t>(), std::vector<size_t>()});
    params[0]->set_friendly_name("start");
    params[1]->set_friendly_name("stop");
    params[2]->set_friendly_name("step");

    auto range = std::make_shared<ngraph::opset3::Range>(params[0], params[1], params[2]);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(range)};
    function = std::make_shared<ngraph::Function>(results, params, "Range");
}

TEST_P(RangeLayerTest, CompareWithRefs) {
    Run();
}

std::string RangeNumpyLayerTest::getTestCaseName(testing::TestParamInfo<RangeParams> obj) {
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision paramPrc;
    InferenceEngine::Precision outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    float start, stop, step;
    std::string targetDevice;
    std::tie(start, stop, step, paramPrc, netPrc, outPrc, inLayout, outLayout, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "paramPRC=" << paramPrc.name() << separator;
    result << "netPRC=" << netPrc.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void RangeNumpyLayerTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    auto blobStart = inferRequest.GetBlob("start");
    blobStart = FuncTestUtils::createAndFillBlobWithFloatArray(blobStart->getTensorDesc(), &start, 1);

    auto blobStop = inferRequest.GetBlob("stop");
    blobStop = FuncTestUtils::createAndFillBlobWithFloatArray(blobStop->getTensorDesc(), &stop, 1);

    auto blobStep = inferRequest.GetBlob("step");
    blobStep = FuncTestUtils::createAndFillBlobWithFloatArray(blobStep->getTensorDesc(), &step, 1);

    inferRequest.Infer();
}

void RangeNumpyLayerTest::SetUp() {
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision paramPrc;
    std::tie(start, stop, step, paramPrc, netPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    auto ngNetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc);
    auto ngParamPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(paramPrc);

    auto params = ngraph::builder::makeParams(ngParamPrc, {std::vector<size_t>(), std::vector<size_t>(), std::vector<size_t>()});
    params[0]->set_friendly_name("start");
    params[1]->set_friendly_name("stop");
    params[2]->set_friendly_name("step");

    auto range = std::make_shared<ngraph::opset4::Range>(params[0], params[1], params[2], ngNetPrc);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(range)};
    function = std::make_shared<ngraph::Function>(results, params, "Range");
}

TEST_P(RangeNumpyLayerTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions