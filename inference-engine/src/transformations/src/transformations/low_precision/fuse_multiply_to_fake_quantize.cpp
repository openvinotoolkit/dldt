﻿// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/fuse_multiply_to_fake_quantize.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FuseMultiplyToFakeQuantizeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

bool FuseMultiplyToFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    const auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    const auto parent = multiply->get_input_node_shared_ptr(0);
    auto fakeQuantize = as_type_ptr<opset1::FakeQuantize>(parent);
    const auto convert = as_type_ptr<opset1::Convert>(parent);

    if (convert) {
        fakeQuantize = as_type_ptr<opset1::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    const auto multiplyConstant = multiply->get_input_node_shared_ptr(1);

    auto outputLowConst = fakeQuantize->get_input_node_shared_ptr(3);
    auto outputHighConst = fakeQuantize->get_input_node_shared_ptr(4);

    const auto value = multiplyConstant->get_output_element_type(0) == outputLowConst->get_output_element_type(0) ?
        multiplyConstant :
        fold<opset1::Convert>(multiplyConstant, outputLowConst->get_output_element_type(0));

    outputLowConst = fold<opset1::Multiply>(outputLowConst, value);
    outputHighConst = fold<opset1::Multiply>(outputHighConst, value);

    const auto fakeQuantizeParent = fakeQuantize->get_input_node_shared_ptr(0);
    const size_t parentIndex = NetworkHelper::getParentOutputIndex(fakeQuantizeParent, fakeQuantize);

    auto newFakeQuantize = std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
        opset1::FakeQuantize(
            fakeQuantizeParent->output(parentIndex),
            fakeQuantize->input_value(1),
            fakeQuantize->input_value(2),
            outputLowConst,
            outputHighConst,
            fakeQuantize->get_levels()),
        multiply->get_output_element_type(0));

    replace_node(multiply, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);

    updateOutput(context, newFakeQuantize, multiply);
    return true;
}

bool FuseMultiplyToFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!is_type<opset1::Constant>(operation->get_input_node_shared_ptr(1))) {
        return false;
    }

    const auto parent = operation->get_input_node_shared_ptr(0);
    auto fq = as_type_ptr<opset1::FakeQuantize>(parent);
    const auto convert = as_type_ptr<opset1::Convert>(parent);

    if (convert) {
        fq = as_type_ptr<opset1::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    if (!fq) {
        return false;
    }

    if (fq->get_output_target_inputs(0).size() != 1) {
        return false;
    }

    return true;
}

bool FuseMultiplyToFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
