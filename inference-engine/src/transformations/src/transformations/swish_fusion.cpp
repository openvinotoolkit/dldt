// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/swish_fusion.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

bool check_constant_value(const std::shared_ptr<ngraph::opset4::Constant>& constant) {
    if (!constant) {
        return false;
    }
    if (constant->get_element_type() == ngraph::element::f32 || constant->get_element_type() == ngraph::element::f16) {
        auto data = constant->cast_vector<float>();
        if (data.size() != 1 || data[0] != 1.0) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

ngraph::pass::SwishFusionWithSigmoid::SwishFusionWithSigmoid() {
    auto input = ngraph::pattern::any_input();
    auto sigmoid = std::make_shared<ngraph::opset4::Sigmoid>(input);
    auto mul = std::make_shared<ngraph::opset4::Multiply>(input, sigmoid);

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto swish = std::make_shared<ngraph::opset4::Swish>(exp_input);

        swish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(sigmoid).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr()},
                                  swish);
        ngraph::replace_node(m.get_match_root(), swish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "SwishWithSigmoidFusion");
    register_matcher(m, matcher_pass_callback);
}

ngraph::pass::SwishFusionWithBeta::SwishFusionWithBeta() {
    auto input = ngraph::pattern::any_input();
    auto beta = ngraph::pattern::any_input();
    auto mul = std::make_shared<ngraph::opset4::Multiply>(input, beta);
    auto neg = std::make_shared<ngraph::opset4::Negative>(mul);
    auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto add = std::make_shared<ngraph::opset4::Add>(exp, add_constant);
    auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto constant = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        if (!check_constant_value(constant)) {
            return false;
        }

        auto swish = std::make_shared<ngraph::opset4::Swish>(exp_input, pattern_to_output.at(beta));

        swish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(beta).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(neg).get_node_shared_ptr(),
                                   pattern_to_output.at(exp).get_node_shared_ptr(),
                                   pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(div).get_node_shared_ptr()},
                                  swish);
        ngraph::replace_node(m.get_match_root(), swish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, "SwishWithBetaFusion");
    register_matcher(m, matcher_pass_callback);
}

ngraph::pass::SwishFusionWithoutBeta::SwishFusionWithoutBeta() {
    auto input = ngraph::pattern::any_input();
    auto neg = std::make_shared<ngraph::opset4::Negative>(input);
    auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto add = std::make_shared<ngraph::opset4::Add>(exp, add_constant);
    auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto & pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto constant = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        if (!check_constant_value(constant)) {
            return false;
        }

        auto swish = std::make_shared<ngraph::opset4::Swish>(exp_input);

        swish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(neg).get_node_shared_ptr(),
                                   pattern_to_output.at(exp).get_node_shared_ptr(),
                                   pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(div).get_node_shared_ptr()},
                                   swish);
        ngraph::replace_node(m.get_match_root(), swish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, "SwishWithoutBetaFusion");
    register_matcher(m, matcher_pass_callback);
}
