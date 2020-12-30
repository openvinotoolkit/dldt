// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/gemm_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
        ngraph::element::f32,
        // ngraph::element::f16
};

const std::vector<ngraph::Shape> dimensions = {
    { 1, 3, 16, 16 }
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setSupportAsymmetricQuantization(true),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setSupportAsymmetricQuantization(false),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, GemmTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(dimensions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    GemmTransformation::getTestCaseName);
}  // namespace




