// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using FuseTransposeAndMatMulParams = std::tuple<
        std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector>,    // Input shape A and Input shape B
        InferenceEngine::Precision                                               // Input precision
>;

class FuseTransposeAndMatMulTest : public testing::WithParamInterface<FuseTransposeAndMatMulParams>, public CPUTestsBase,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseTransposeAndMatMulParams> obj);

protected:
    void SetUp() override;
    virtual void CreateGraph();
    void CheckTransposeCount(size_t expectedTransposeCount);

    InferenceEngine::SizeVector inputShapeA;
    InferenceEngine::SizeVector inputShapeB;
    InferenceEngine::Precision inPrec;
};

class FuseTransposeAndMatMulTest1 : public FuseTransposeAndMatMulTest {
protected:
    void CreateGraph() override;
};

class FuseTransposeAndMatMulTest2 : public FuseTransposeAndMatMulTest {
protected:
    void CreateGraph() override;
};

} // namespace SubgraphTestsDefinitions
