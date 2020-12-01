// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_nonzero.hpp"
#include "vpu/private_plugin_config.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(ImportNonZero, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
    {}
};

const std::vector<std::map<std::string, std::string>> importConfigs = {
    {}
};

INSTANTIATE_TEST_CASE_P(smoke_ImportNetworkCase, ImportNonZero,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::ValuesIn(exportConfigs),
                                ::testing::ValuesIn(importConfigs)),
                        ImportNonZero::getTestCaseName);

} // namespace
