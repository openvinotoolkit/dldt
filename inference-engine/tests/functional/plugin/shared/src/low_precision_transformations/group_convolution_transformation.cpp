// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/group_convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "lpt_ngraph_functions/group_convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string GroupConvolutionTransformation::getTestCaseName(testing::TestParamInfo<GroupConvolutionTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionTransformationParam param;
    bool addPrecisionPreserved;
    std::tie(netPrecision, targetDevice, params, param, addPrecisionPreserved) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, param.inputShape, targetDevice, params) << "_" <<
        param.inputShape << "_" <<
        param.outputShape << "_" <<
        param.group << "_" <<
        param.fakeQuantizeOnData << "_" <<
        (addPrecisionPreserved ? "max_pool_" : "") <<
        param.fakeQuantizeOnWeights;
    return result.str();
}

void GroupConvolutionTransformation::SetUp() {
    threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    GroupConvolutionTransformationParam param;
    bool addPrecisionPreserved;
    std::tie(netPrecision, targetDevice, params, param, addPrecisionPreserved) = this->GetParam();

    function = ngraph::builder::subgraph::GroupConvolutionFunction::getOriginal(
        netPrecision,
        param.inputShape,
        param.outputShape,
        param.group,
        param.fakeQuantizeOnData,
        param.fakeQuantizeOnWeights,
        addPrecisionPreserved);
}

TEST_P(GroupConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
