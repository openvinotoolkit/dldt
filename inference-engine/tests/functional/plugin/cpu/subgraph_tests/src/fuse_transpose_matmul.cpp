// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/fuse_transpose_matmul.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

std::string FuseTransposeAndMatMulTest::getTestCaseName(testing::TestParamInfo<FuseTransposeAndMatMulParams> obj) {
    std::ostringstream result;
    std::tuple<SizeVector, SizeVector> inputShapes;
    SizeVector inputShapeA, inputShapeB;
    Precision inPrec;
    std::tie(inputShapes, inPrec) = obj.param;
    std::tie(inputShapeA, inputShapeB) = inputShapes;

    result << "IS_A=" << CommonTestUtils::vec2str(inputShapeA) << "_";
    result << "IS_B=" << CommonTestUtils::vec2str(inputShapeB) << "_";
    result << "Precision=" << inPrec.name();

    return result.str();
}

void FuseTransposeAndMatMulTest::CheckTransposeCount(size_t expectedTransposeCount) {
    InferenceEngine::CNNNetwork execGraphInfo = executableNetwork.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    size_t actualTransposeCount = 0;
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);
            return value->get();
        };
        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Transpose") {
            actualTransposeCount++;
        }
    }

    ASSERT_EQ(expectedTransposeCount, actualTransposeCount);
}

void FuseTransposeAndMatMulTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    std::tuple<SizeVector, SizeVector> inputShapes;
    std::tie(inputShapes, inPrec) = this->GetParam();
    std::tie(inputShapeA, inputShapeB) = inputShapes;

    CreateGraph();
}

/*  FuseTransposeAndMatMulTest graph
      ---------
      | Input |
      ---------
          |
    -------------       ---------
    | Transpose |       | Input |
    -------------       ---------
          |                 |
          |    ----------   |
          |----| MatMul |---|
               ----------
                    |
               ----------
               | Output |
               ----------
*/

void FuseTransposeAndMatMulTest::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, { inputShapeA, inputShapeB });

    std::vector<int64_t> order(inputShapeA.size());
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[order.size() - 1], order[order.size() - 2]);

    auto constOrderA = ngraph::builder::makeConstant(ngraph::element::i64, {inputShapeA.size()}, order);
    auto transposeA = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrderA);
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(transposeA, params[1]);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(matmul)};
    function = std::make_shared<ngraph::Function>(results, params, "TransposeMatMul");
}

TEST_P(FuseTransposeAndMatMulTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckTransposeCount(0);
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::tuple<SizeVector, SizeVector>> inputShapes = {
        std::tuple<SizeVector, SizeVector>{{1, 10}, {1, 10}},
        std::tuple<SizeVector, SizeVector>{{5, 10}, {5, 1}},
        std::tuple<SizeVector, SizeVector>{{1, 8, 10}, {1, 8, 15}},
        std::tuple<SizeVector, SizeVector>{{2, 5, 10, 15}, {1, 5, 10, 5}},
};

const auto fuseTransposeAndMatMulParams = ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(netPrecisions)
);

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndMatMulTest, fuseTransposeAndMatMulParams, FuseTransposeAndMatMulTest::getTestCaseName);


/*  FuseTransposeAndMatMulTest1 graph
                        ---------
                        | Input |
                        ---------
                            |
      ---------       -------------
      | Input |       | Transpose |
      ---------       -------------
          |                 |
          |    ----------   |
          |----| MatMul |---|
               ----------
                    |
               ----------
               | Output |
               ----------
*/

void FuseTransposeAndMatMulTest1::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, { inputShapeA, inputShapeB });

    std::vector<int64_t> order(inputShapeB.size());
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[order.size() - 1], order[order.size() - 2]);

    auto constOrderB = ngraph::builder::makeConstant(ngraph::element::i64, {inputShapeB.size()}, order);
    auto transposeB = std::make_shared<ngraph::opset5::Transpose>(params[1], constOrderB);
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(params[0], transposeB);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(matmul)};
    function = std::make_shared<ngraph::Function>(results, params, "TransposeMatMul");
}

TEST_P(FuseTransposeAndMatMulTest1, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckTransposeCount(0);
}

const std::vector<std::tuple<SizeVector, SizeVector>> inputShapes1 = {
        std::tuple<SizeVector, SizeVector>{{1, 10}, {1, 10}},
        std::tuple<SizeVector, SizeVector>{{5, 10}, {1, 10}},
        std::tuple<SizeVector, SizeVector>{{1, 8, 10}, {1, 15, 10}},
        std::tuple<SizeVector, SizeVector>{{2, 5, 10, 15}, {1, 5, 7, 15}},
};

const auto fuseTransposeAndMatMulParams1 = ::testing::Combine(
        ::testing::ValuesIn(inputShapes1),
        ::testing::ValuesIn(netPrecisions)
);

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndMatMulTest1, fuseTransposeAndMatMulParams1, FuseTransposeAndMatMulTest::getTestCaseName);


/*  FuseTransposeAndMatMulTest2 graph
      ---------         ---------
      | Input |         | Input |
      ---------         ---------
          |                 |
    -------------     -------------
    | Transpose |     | Transpose |
    -------------     -------------
          |                 |
          |    ----------   |
          |----| MatMul |---|
               ----------
                    |
               ----------
               | Output |
               ----------
*/

void FuseTransposeAndMatMulTest2::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, { inputShapeA, inputShapeB });

    std::vector<int64_t> order(inputShapeA.size());
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[order.size() - 1], order[order.size() - 2]);

    auto constOrderA = ngraph::builder::makeConstant(ngraph::element::i64, {inputShapeA.size()}, order);
    auto constOrderB = ngraph::builder::makeConstant(ngraph::element::i64, {inputShapeB.size()}, order);
    auto transposeA = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrderA);
    auto transposeB = std::make_shared<ngraph::opset5::Transpose>(params[1], constOrderB);
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(transposeA, transposeB);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(matmul)};
    function = std::make_shared<ngraph::Function>(results, params, "TransposeMatMul");
 }


TEST_P(FuseTransposeAndMatMulTest2, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckTransposeCount(0);
}

const std::vector<std::tuple<SizeVector, SizeVector>> inputShapes2 = {
        std::tuple<SizeVector, SizeVector>{{1, 10}, {10, 1}},
        std::tuple<SizeVector, SizeVector>{{5, 10}, {8, 5}},
        std::tuple<SizeVector, SizeVector>{{1, 15, 8}, {1, 2, 15}},
        std::tuple<SizeVector, SizeVector>{{2, 5, 10, 3}, {1, 5, 8, 10}},
};

const auto fuseTransposeAndMatMulParams2 = ::testing::Combine(
         ::testing::ValuesIn(inputShapes2),
         ::testing::ValuesIn(netPrecisions)
);

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndMatMulTest2, fuseTransposeAndMatMulParams2, FuseTransposeAndMatMulTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions
