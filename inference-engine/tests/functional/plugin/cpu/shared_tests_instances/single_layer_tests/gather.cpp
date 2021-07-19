// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

const std::vector<std::vector<size_t>> inputShapes_1D = {
        std::vector<size_t>{4},
};

const std::vector<std::vector<size_t>> indicesShapes_1D = {
        std::vector<size_t>{1},
        std::vector<size_t>{3},
};

const std::vector<std::tuple<int, int>> axes_batchdims_1D = {
        std::tuple<int, int>{0, 0}
};

const auto gather7Params_1D = testing::Combine(
        testing::ValuesIn(inputShapes_1D),
        testing::ValuesIn(indicesShapes_1D),
        testing::ValuesIn(axes_batchdims_1D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_1D, Gather7LayerTest, gather7Params_1D, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes_2D = {
        std::vector<size_t>{4, 19},
};

const std::vector<std::vector<size_t>> indicesShapes_2D = {
        std::vector<size_t>{4},
        std::vector<size_t>{4, 2},
};

const std::vector<std::tuple<int, int>> axes_batchdims_2D = {
        std::tuple<int, int>{0, 0},
        std::tuple<int, int>{1, 0},
        std::tuple<int, int>{1, 1},
        std::tuple<int, int>{-1, -1},
};

const auto gather7Params_2D = testing::Combine(
        testing::ValuesIn(inputShapes_2D),
        testing::ValuesIn(indicesShapes_2D),
        testing::ValuesIn(axes_batchdims_2D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_2D, Gather7LayerTest, gather7Params_2D, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes4D = {
        std::vector<size_t>{4, 5, 6, 7},
};

const std::vector<std::vector<size_t>> indicesShapes_BD0 = {
        std::vector<size_t>{4},
        std::vector<size_t>{2, 2},
        std::vector<size_t>{3, 2, 4},
};

const std::vector<std::tuple<int, int>> axes_BD0 = {
        std::tuple<int, int>{0, 0},
        std::tuple<int, int>{1, 0},
        std::tuple<int, int>{2, 0},
        std::tuple<int, int>{-1, 0},
};

const auto gather7ParamsSubset_BD0 = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_BD0),
        testing::ValuesIn(axes_BD0),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_BD0, Gather7LayerTest, gather7ParamsSubset_BD0, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> indicesShapes_BD1 = {
        std::vector<size_t>{4, 2},
        std::vector<size_t>{4, 5, 3},
        std::vector<size_t>{4, 1, 2, 3},
};

const std::vector<std::tuple<int, int>> axes_BD1 = {
        std::tuple<int, int>{1, 1},
        std::tuple<int, int>{2, 1},
        std::tuple<int, int>{-1, 1},
        std::tuple<int, int>{-2, 1},
};

const auto gather7ParamsSubset_BD1 = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_BD1),
        testing::ValuesIn(axes_BD1),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_BD1, Gather7LayerTest, gather7ParamsSubset_BD1, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> indicesShapes_BD2 = {
        std::vector<size_t>{4, 5, 4, 3},
        std::vector<size_t>{4, 5, 3, 2}
};

const std::vector<std::tuple<int, int>> axes_BD2 = {
        std::tuple<int, int>{2, 2},
        std::tuple<int, int>{3, -2},
        std::tuple<int, int>{-1, 2},
        std::tuple<int, int>{-1, -2},
};

const auto gather7ParamsSubset_BD2 = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_BD2),
        testing::ValuesIn(axes_BD2),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_BD2, Gather7LayerTest, gather7ParamsSubset_BD2, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> indicesShapes_NegativeBD = {
        std::vector<size_t>{4, 5, 4},
        std::vector<size_t>{4, 5, 3}
};

const std::vector<std::tuple<int, int>> axes_NegativeBD = {
        std::tuple<int, int>{0, -3},
        std::tuple<int, int>{1, -2},
        std::tuple<int, int>{2, -2},
        std::tuple<int, int>{-2, -2},
        std::tuple<int, int>{-1, -1},
        std::tuple<int, int>{-2, -1},
};

const auto gather7ParamsSubset_NegativeBD = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_NegativeBD),
        testing::ValuesIn(axes_NegativeBD),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_NegativeBD, Gather7LayerTest, gather7ParamsSubset_NegativeBD, Gather7LayerTest::getTestCaseName);

const auto gatherParamsVec = testing::Combine(
        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 30, 50, 1}})),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 16, 16}, {10, 7, 8}, {10, 5, 7}, {10, 5}})),
        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{2, 1}}),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_Vec, Gather7LayerTest, gatherParamsVec, Gather7LayerTest::getTestCaseName);

//const auto gatherParamsTemp11 = testing::Combine(
//        testing::ValuesIn(std::vector<std::vector<size_t>>({{6, 10, 10, 1}})),
//        testing::ValuesIn(std::vector<std::vector<size_t>>({{6, 8, 10}})),
//        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{2, 1}}),
//        testing::ValuesIn(netPrecisions),
//        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//        testing::Values(InferenceEngine::Layout::ANY),
//        testing::Values(InferenceEngine::Layout::ANY),
//        testing::Values(CommonTestUtils::DEVICE_CPU)
//);
//
//INSTANTIATE_TEST_CASE_P(smoke_Gather_long2, Gather7LayerTest, gatherParamsTemp11, Gather7LayerTest::getTestCaseName);

//const auto gatherParamsShort1 = testing::Combine(
//        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 100, 100, 20, 2, 1}})),
//        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 5}})),
//        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{4, 1}}),
//        testing::ValuesIn(netPrecisions),
//        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//        testing::Values(InferenceEngine::Layout::ANY),
//        testing::Values(InferenceEngine::Layout::ANY),
//        testing::Values(CommonTestUtils::DEVICE_CPU)
//);
//
//INSTANTIATE_TEST_CASE_P(smoke_Gather_short1, Gather7LayerTest, gatherParamsShort1, Gather7LayerTest::getTestCaseName);

//const auto gatherParamsShort2 = testing::Combine(
//        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 30, 50, 2, 1}})),
//        testing::ValuesIn(std::vector<std::vector<size_t>>({{10, 5}})),
//        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{3, 1}}),
//        testing::ValuesIn(netPrecisions),
//        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//        testing::Values(InferenceEngine::Layout::ANY),
//        testing::Values(InferenceEngine::Layout::ANY),
//        testing::Values(CommonTestUtils::DEVICE_CPU)
//);
//
//INSTANTIATE_TEST_CASE_P(smoke_Gather_short2, Gather7LayerTest, gatherParamsShort2, Gather7LayerTest::getTestCaseName);

}  // namespace
