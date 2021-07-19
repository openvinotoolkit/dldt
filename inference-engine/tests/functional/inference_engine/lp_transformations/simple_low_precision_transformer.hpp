// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <ngraph/ngraph.hpp>

#include "layer_transformation.hpp"
#include "common_test_utils/test_common.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/common/operation_precision_restriction.hpp"
#include "low_precision/common/operation_per_tensor_quantization_restriction.hpp"

class SimpleLowPrecisionTransformer : public ngraph::pass::FunctionPass{
public:
    SimpleLowPrecisionTransformer(
        const std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>& precisionRestrictions = {},
        const std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>& quantizationRestrictions = {});

    template <class T, class Operation>
    void add(const TestTransformationParams& params) {
        commonGraphRewrite->add_matcher<T>(TestTransformationParams::toParams(params));
    }

    void transform(std::shared_ptr<ngraph::Function>& function);
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    std::shared_ptr<ngraph::pass::Manager> markup;
    std::shared_ptr<ngraph::pass::Manager> common;
    std::shared_ptr<ngraph::pass::GraphRewrite> commonGraphRewrite;
    std::shared_ptr<ngraph::pass::GraphRewrite> cleanup;
};
