// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather_tree.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32
};

const std::vector<std::vector<size_t>> inputShapes = {{1, 1, 10}};/**/

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER
};

INSTANTIATE_TEST_CASE_P(Basic_smoke, GatherTreeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapes),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        GatherTreeLayerTest::getTestCaseName);

}  // namespace
