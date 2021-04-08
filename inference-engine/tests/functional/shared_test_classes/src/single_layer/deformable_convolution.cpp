// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/deformable_convolution.hpp"

namespace LayerTestsDefinitions {

std::string DeformableConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<deformableConvLayerTestParamsSet> obj) {
    deformableConvSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) =
        obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector offsets, filter, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    std::tie(offsets, filter, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "DV" << CommonTestUtils::vec2str(offsets) << "_";
    result << "K" << CommonTestUtils::vec2str(filter) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "G=" << groups << "_";
    result << "DG=" << deformable_groups << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr DeformableConvolutionLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
        InferenceEngine::Blob::Ptr blobPtr;
        const std::string& name = info.name();
        if (name == "a_data") {
            blobPtr = LayerTestsUtils::LayerTestsCommon::GenerateInput(info);
        } else if (name == "b_offset_vals") {
            blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 0, 0.1);
        } else if (name == "c_filter_vals") {
            blobPtr = LayerTestsUtils::LayerTestsCommon::GenerateInput(info);
        }
        return blobPtr;
}

void DeformableConvolutionLayerTest::SetUp() {
    deformableConvSpecificParams convParams;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
        this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector offsets, filter, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    std::tie(offsets, filter, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, offsets, filter});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto data = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(inputShape));
    data->set_friendly_name("a_data");
    auto offset_vals = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(offsets));
    offset_vals->set_friendly_name("b_offset_vals");
    auto filter_vals = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(filter));
    filter_vals->set_friendly_name("c_filter_vals");
    auto deformable_conv = std::make_shared<ngraph::opset1::DeformableConvolution>(data, offset_vals, filter_vals,
                                                              stride, padBegin, padEnd, dilation, padType, groups, deformable_groups);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(deformable_conv)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{data, offset_vals, filter_vals}, "deformable_convolution");
}
}  // namespace LayerTestsDefinitions
