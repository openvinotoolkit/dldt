// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request_callback.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {{{}}}
};

const std::vector<std::string> devices{CommonTestUtils::DEVICE_CPU};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CallbackTests,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::ValuesIn(devices),
            ::testing::ValuesIn(configs)),
        CallbackTests::getTestCaseName);
}  // namespace
