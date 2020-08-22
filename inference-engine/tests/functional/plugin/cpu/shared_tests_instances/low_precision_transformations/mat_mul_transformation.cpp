// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "low_precision_transformations/mat_mul_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versionValues = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

std::vector<MatMulTransformationTestValues> testValues = {
    {
        { 1, 4, 12, 2 },
        { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        { 1, 4, 2, 12 },
        { 256ul, ngraph::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} }
    },
    {
        { 8, 4, 12, 2 },
        { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        { 8, 4, 2, 12 },
        { 256ul, ngraph::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} }
    },
    {
        { 1, 4, 12, 2 },
        { 256ul, ngraph::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
        { 1, 4, 2, 12 },
        { 256ul, ngraph::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} }
    }
};

INSTANTIATE_TEST_CASE_P(LPT, MatMulTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 384, 1024 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(versionValues),
        ::testing::ValuesIn(testValues)),
    MatMulTransformation::getTestCaseName);
}  // namespace
