// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp"

#include <memory>
#include <vector>
#include <string>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/topk_ie.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertTopKToTopKIE::convert_topk_to_topk_ie() {
    auto topk = std::make_shared<pattern::op::Label>(element::f32, Shape{1}, pattern::has_class<opset1::TopK>());

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto topk = std::dynamic_pointer_cast<opset1::TopK>(m.get_match_root());
        if (!topk || topk->input(1).get_partial_shape().rank().is_dynamic()) {
            return false;
        }
        if (topk->input(1).get_partial_shape().rank().get_length() == 1) {
            return false;
        }

        // WA: if we replace TopK second input with Unsqueeze operation we will get dynamic shape until first CF pass
        // but due to not all legacy operations support dynamic input shapes and dynamic shape can break pipeline we
        // need to unsqueeze constant manually.
        Output<Node> unsqueezed_k;
        NodeVector new_ops;
        if (auto k_const = std::dynamic_pointer_cast<opset1::Constant>(topk->input_value(1).get_node_shared_ptr())) {
            auto k_value = k_const->cast_vector<int64_t>();
            unsqueezed_k = opset1::Constant::create(element::i64, Shape{1}, k_value);
        } else {
            unsqueezed_k = std::make_shared<opset1::Unsqueeze>(topk->input_value(1), opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(unsqueezed_k.get_node_shared_ptr());
        }

        auto topk_ie = std::make_shared<ngraph::op::TopKIE>(topk->input_value(0), unsqueezed_k, topk->get_axis(), topk->get_mode(),
                                                             topk->get_sort_type());
        new_ops.push_back(topk_ie);

        Output<Node> index_output;
        // insert Convert if index element type not equal to i32
        if (topk->get_index_element_type() == element::i32) {
            index_output = topk_ie->output(1);
        } else {
            index_output = std::make_shared<opset1::Convert>(topk_ie->output(1), topk->get_index_element_type());
            new_ops.push_back(index_output.get_node_shared_ptr());
        }

        topk_ie->set_friendly_name(topk->get_friendly_name());
        ngraph::copy_runtime_info(topk, {unsqueezed_k.get_node_shared_ptr(), topk_ie});
        topk->output(0).replace(topk_ie->output(0));
        topk->output(1).replace(index_output);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(topk, "ConvertTopKToTopKIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}