// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior2/infer_request/io_blob.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                ::testing::Values(std::map<std::string, std::string>({}))),
        InferRequestIOBBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestIOBBlobTest,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(configs)),
        InferRequestIOBBlobTest::getTestCaseName);
}  // namespace
