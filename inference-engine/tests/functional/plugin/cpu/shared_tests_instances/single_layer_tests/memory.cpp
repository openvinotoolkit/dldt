// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/memory.h"

using namespace LayerTestsDefinitions;

namespace {

std::vector<ngraph::op::MemoryTransformation> transformation {
        ngraph::op::MemoryTransformation::NONE,
        ngraph::op::MemoryTransformation::LOW_LATENCY_V2,
        ngraph::op::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API,
        ngraph::op::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT,
};

const std::vector<InferenceEngine::SizeVector> inShapes = {
        {3},
        {100, 100},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<int64_t> iterationCount {
        1,
        3,
        10
};

INSTANTIATE_TEST_CASE_P(smoke_MemoryTest, MemoryTest,
        ::testing::Combine(
                ::testing::ValuesIn(transformation),
                ::testing::ValuesIn(iterationCount),
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        MemoryTest::getTestCaseName);

} // namespace

