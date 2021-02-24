// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/group_convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(GroupConvolutionLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> precisions = {
    InferenceEngine::Precision::FP64, InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16, InferenceEngine::Precision::BF16
};
const std::vector<std::vector<size_t>> kernels = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<size_t> numGroups = {2, 8};
const std::vector<ngraph::op::PadType> pad_types = {
    ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID,
    ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER};
const auto inputShapes = std::vector<size_t>({1, 16, 30, 30});

const auto groupConv2DParams = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::ValuesIn(numGroups), ::testing::ValuesIn(pad_types));

INSTANTIATE_TEST_CASE_P(
    smoke_GroupConvolution2D_Serialization, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv2DParams, ::testing::ValuesIn(precisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(inputShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

}  // namespace
