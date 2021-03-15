// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_min.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ReduceMinTransformation::ReduceMinTransformation(const Params& params) : ReduceBaseTransformation(params) {}

void ReduceMinTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::ReduceMin>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

bool ReduceMinTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    return is_type<opset1::ReduceMin>(reduce) ? ReduceBaseTransformation::canBeTransformed(context, reduce) : false;
}

bool ReduceMinTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return true;
}

bool ReduceMinTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return true;
}

bool ReduceMinTransformation::getKeepDims(const std::shared_ptr<Node>& reduce) const {
    return as_type_ptr<opset1::ReduceMin>(reduce)->get_keep_dims();
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
