// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ie_core.hpp>
#include <ngraph_functions/builders.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "single_layer_tests/scatter_ND_update.hpp"

using namespace ngraph::opset3;

namespace LayerTestsDefinitions {

std::string ScatterNDUpdateLayerTest::getTestCaseName(const testing::TestParamInfo<scatterNDUpdateParamsTuple> &obj) {
    sliceSelcetInShape shapeDescript;
    std::vector<size_t> inShape;
    std::vector<size_t> indicesShape;
    std::vector<size_t> indicesValue;
    std::vector<size_t> updateShape;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::string targetName;
    std::tie(shapeDescript, inputPrecision, indicesPrecision, targetName) = obj.param;
    std::tie(inShape, indicesShape, indicesValue, updateShape) = shapeDescript;
    std::ostringstream result;
    result << "InputShape=" << CommonTestUtils::vec2str(inShape) << "_";
    result << "IndicesShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
    result << "UpdateShape=" << CommonTestUtils::vec2str(updateShape) << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "idxPrc=" << indicesPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

std::vector<sliceSelcetInShape> ScatterNDUpdateLayerTest::combineShapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>>& inputShapes) {
    std::vector<sliceSelcetInShape> resVec;
    for (auto& inputShape : inputShapes) {
        for (auto& item : inputShape.second) {
            auto indiceShape = item.first;
            size_t indicesRank = indiceShape.size();
            std::vector<size_t> updateShape;
            for (size_t i = 0; i < indicesRank - 1; i++) {
                updateShape.push_back(indiceShape[i]);
            }
            auto srcShape = inputShape.first;
            for (size_t j = indiceShape[indicesRank - 1]; j < srcShape.size(); j++) {
                updateShape.push_back(srcShape[j]);
            }
            resVec.push_back(std::make_tuple(srcShape, indiceShape, item.second, updateShape));
        }
    }
    return resVec;
}

void ScatterNDUpdateLayerTest::SetUp() {
    sliceSelcetInShape shapeDescript;
    InferenceEngine::SizeVector inShape;
    InferenceEngine::SizeVector indicesShape;
    InferenceEngine::SizeVector indicesValue;
    InferenceEngine::SizeVector updateShape;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::tie(shapeDescript, inputPrecision, indicesPrecision, targetDevice) = this->GetParam();
    std::tie(inShape, indicesShape, indicesValue, updateShape) = shapeDescript;
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto idxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indicesPrecision);
    ngraph::ParameterVector paramVector;
    auto inputParams = std::make_shared<ngraph::opset1::Parameter>(inPrc, ngraph::Shape(inShape));
    paramVector.push_back(inputParams);
    auto updateParams = std::make_shared<ngraph::opset1::Parameter>(inPrc, ngraph::Shape(updateShape));
    paramVector.push_back(updateParams);
    auto paramVectorOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramVector));
    auto s2d = ngraph::builder::makeScatterNDUpdate(paramVectorOuts[0], idxPrc, indicesShape, indicesValue, paramVectorOuts[1]);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2d)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "ScatterNDUpdate");
}

TEST_P(ScatterNDUpdateLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions