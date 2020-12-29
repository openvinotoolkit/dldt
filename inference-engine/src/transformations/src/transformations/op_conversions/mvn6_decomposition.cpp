// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/mvn6_decomposition.hpp"

#include <memory>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MVN6Decomposition, "MVN6Decomposition", 0);

ngraph::pass::MVN6Decomposition::MVN6Decomposition() {
    // Decomposes MVN(x, axes) op if normalize_variance is false into sub-graph
    // x - ReduceMean(x, axes), if normalize_variance is true into sub-graph
    // (x - ReduceMean(x, axes)) / Sqrt(ReduceSum((x - ReduceMean(x, axes)) ^ 2))
    auto mvn = ngraph::pattern::wrap_type<opset6::MVN>();

    ngraph::matcher_pass_callback callback = [&](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto match_root = m.get_match_root();
        auto mvn_node = std::dynamic_pointer_cast<ngraph::opset6::MVN>(match_root);

        if (mvn_node == nullptr || m_transformation_callback(mvn)) {
            return false;
        }

        const auto data = mvn_node->input_value(0);
        const auto axes = mvn_node->input_value(1);

        auto mean = std::make_shared<ngraph::opset6::ReduceMean>(data, axes, true);
        auto mean_normalization = std::make_shared<ngraph::opset6::Subtract>(data, mean);

        if (!mvn_node->get_normalize_variance()) {
            mean_normalization->set_friendly_name(match_root->get_friendly_name());
            ngraph::copy_runtime_info(mvn_node, { mean, mean_normalization });
            ngraph::replace_node(match_root, mean_normalization);
        } else {
            auto mul = std::make_shared<ngraph::opset6::Multiply>(mean_normalization, mean_normalization);
            auto sum = std::make_shared<ngraph::opset6::ReduceSum>(mul, axes, true);

            auto eps = mvn_node->get_eps();
            auto eps_node = ngraph::opset6::Constant::create(data.get_element_type(), ngraph::Shape{ 1 }, { eps });
            auto eps_mode = mvn_node->get_eps_mode();

            std::shared_ptr<ngraph::opset6::Add> eps_add;
            std::shared_ptr<ngraph::opset6::Sqrt> sqrt;
            std::shared_ptr<ngraph::opset6::Divide> div;

            if (eps_mode == op::MVNEpsMode::INSIDE_SQRT) {
                eps_add = std::make_shared<ngraph::opset6::Add>(sum, eps_node);
                sqrt = std::make_shared<ngraph::opset6::Sqrt>(eps_add);
                div = std::make_shared<ngraph::opset6::Divide>(mean_normalization, sqrt);
            } else {
                sqrt = std::make_shared<ngraph::opset6::Sqrt>(sum);
                eps_add = std::make_shared<ngraph::opset6::Add>(sqrt, eps_node);
                div = std::make_shared<ngraph::opset6::Divide>(mean_normalization, sqrt);
            }

            div->set_friendly_name(match_root->get_friendly_name());
            ngraph::copy_runtime_info(mvn_node, { mean, mean_normalization, mul, sum, eps_node, eps_add, sqrt, div });
            ngraph::replace_node(match_root, div);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mvn, "MVN6Decomposition");
    register_matcher(m, callback);
}
