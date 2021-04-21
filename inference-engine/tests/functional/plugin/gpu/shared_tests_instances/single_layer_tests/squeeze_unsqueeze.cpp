// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/squeeze_unsqueeze.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ShapeAxesVec> axesVectors = {
        {{4, 2, 1, 3}},
        {{4, 2, 1, 3}, {2}},
        {{1, 1, 1, 1}},
        {{1, 1, 1, 1}, {-1}},
        {{1, 1, 1, 1}, {0}},
        {{1, 1, 1, 1}, {1}},
        {{1, 1, 1, 1}, {2}},
        {{1, 1, 1, 1}, {3}},
        {{1, 1, 1, 1}, {0, 1}},
        {{1, 1, 1, 1}, {0, 2}},
        {{1, 1, 1, 1}, {0, 3}},
        {{1, 1, 1, 1}, {1, 2}},
        {{1, 1, 1, 1}, {2, 3}},
        {{1, 1, 1, 1}, {0, 1, 2}},
        {{1, 1, 1, 1}, {0, 2, 3}},
        {{1, 1, 1, 1}, {1, 2, 3}},
        {{1, 1, 1, 1}, {0, 1, 2, 3}},
        {{1, 2, 3, 4}, {0}},
        {{2, 1, 3, 4}},
        {{2, 1, 3, 4}, {1}},
        {{1}},
        {{1}, {0}},
        {{1}, {-1}},
        {{1, 2}},
        {{1, 2}, {0}},
        {{2, 1}},
        {{2, 1}, {1}},
        {{2, 1}, {-1}},
    };

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<ngraph::helpers::SqueezeOpType> opTypes = {
        ngraph::helpers::SqueezeOpType::SQUEEZE,
        ngraph::helpers::SqueezeOpType::UNSQUEEZE
};

INSTANTIATE_TEST_CASE_P(smoke_Basic, SqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axesVectors),
                                ::testing::ValuesIn(opTypes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);
}  // namespace
