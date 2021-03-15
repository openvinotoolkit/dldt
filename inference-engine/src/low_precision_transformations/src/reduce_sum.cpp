// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_sum.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ReduceSumTransformation::ReduceSumTransformation(const Params& params) : ReduceBaseTransformation(params) {}

void ReduceSumTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::ReduceSum>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

bool ReduceSumTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    return is_type<opset1::ReduceSum>(reduce) ? ReduceBaseTransformation::canBeTransformed(context, reduce) : false;
}

void ReduceSumTransformation::changeDequantizationValues(
    FakeQuantizeDequantization& dequantization,
    const std::shared_ptr<Node>& reduce) const {
    ReduceBaseTransformation::changeDequantizationValues(dequantization, reduce);

    if (dequantization.subtract) {
        const auto reduceSum = as_type_ptr<opset1::ReduceSum>(reduce);
        const auto reductionAxes = reduceSum->get_reduction_axes();
        const auto inputShape = reduceSum->get_input_shape(0);

        // calculating the number of reduced elements
        size_t reductionSize = 1ul;
        for (const auto& elem : reductionAxes) {
            reductionSize *= inputShape[elem];
        }

        // (a1 - s) + (a2 - s) + ... + (an - s) = (a1 + a2 + ... + an) - n * s
        const auto reductionSizeConstant = opset1::Constant::create(deqPrecision, Shape{}, { static_cast<float>(reductionSize) });
        const auto result = fold<opset1::Multiply>(dequantization.subtractConstant, reductionSizeConstant);

        replace_node(dequantization.subtractConstant, result);
        dequantization.subtractConstant = as_type_ptr<opset1::Constant>(result);
    }
}

bool ReduceSumTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return false;
}

bool ReduceSumTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return false;
}

bool ReduceSumTransformation::getKeepDims(const std::shared_ptr<Node>& reduce) const {
    return as_type_ptr<opset1::ReduceSum>(reduce)->get_keep_dims();
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
