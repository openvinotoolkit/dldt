// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
namespace MatMulParams {
enum class InputLayerType {
    CONSTANT,
    PARAMETER,
};
} // namespace MatMulParams
}  // namespace LayerTestsDefinitions

typedef std::tuple<
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        LayerTestsDefinitions::MatMulParams::InputLayerType,
        LayerTestsUtils::TargetDevice
> MatMulLayerTestParamsSet;

namespace LayerTestsDefinitions {

class MatMulTest : public testing::WithParamInterface<MatMulLayerTestParamsSet>, public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
