﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/group_convolution.hpp"

#include <memory>
#include <string>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

GroupConvolutionTransformation::GroupConvolutionTransformation(const Params& params) : ConvolutionTransformation(params) {
    auto matcher = ngraph::pattern::wrap_type<opset1::GroupConvolution>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || m_transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "GroupConvolutionTransformation");
    this->register_matcher(m, callback);
}

void GroupConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    // question to nGraph: why it doesn't work
    // addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::GroupConvolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>()}));

    addSingleNodePattern<opset1::GroupConvolution>(pass, context);
}

bool GroupConvolutionTransformation::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    return WeightableLayerTransformation::isQuantized(layer, true);
}

bool GroupConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolution = m.get_match_root();

    if (!GroupConvolutionTransformation::canBeTransformed(context, convolution)) {
        return false;
    }

    ConvolutionTransformation::transform(context, m);
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
