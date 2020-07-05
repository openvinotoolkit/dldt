// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/lin_op_sequence_fusoin.hpp"
#include "transformations/mul_add_squence_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

template <class T>
Output<Node> eltwise_fold(const Output<Node> & input0, const Output<Node> & input1) {
    auto eltwise = std::make_shared<T>(input0, input1);
    OutputVector output(eltwise->get_output_size());
    if (!eltwise->constant_fold(output, {input0, input1})) {
        throw ngraph_error("Can not constant fold eltwise node");
    }
    if (output.size() != 1) {
        throw ngraph_error("Eltwise constant fold has unexpected number of outputs: " + std::to_string(output.size()));
    }
    return output[0];
}

ngraph::pass::AddMultiplyFusion::AddMultiplyFusion() {
    // Create Add->Multiply pattern where Add has exactly one consumer
    auto m_data = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto m_add_constant = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Constant>());
    auto m_add = std::make_shared<pattern::op::Any>(element::f32, Shape{},
            [](std::shared_ptr<Node> node) {
                // Check that node has type opset3::Add and node has only one consumer
                return std::dynamic_pointer_cast<opset3::Add>(node) && node->output(0).get_target_inputs().size() == 1;
            }, NodeVector{m_data, m_add_constant});
    auto m_mul_constant = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Constant>());
    auto m_mul = std::make_shared<ngraph::opset3::Multiply>(m_add, m_mul_constant);

    ngraph::graph_rewrite_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        auto mul = m.get_match_root();

        auto & label_to_output = m.get_pattern_value_map();
        Output<Node> input = label_to_output[m_data];
        Output<Node> mul_const = label_to_output[m_mul_constant];
        Output<Node> add_const = label_to_output[m_add_constant];

        // Replace Add->Multiply with Multiply->Add
        // As new Multiply can be fused with operation above it we add this Multiply
        // to the list of operations that will be used in additional matching.
        auto new_mul = register_new_node<opset3::Multiply>(input, mul_const);

        // Add two constants using opset3::Add constant folding and create new Add operation
        auto new_add = std::make_shared<opset3::Add>(new_mul, eltwise_fold<opset3::Multiply>(add_const, mul_const));

        copy_runtime_info(mul, {new_mul, new_add});
        new_add->set_friendly_name(mul->get_friendly_name());
        replace_node(mul, new_add);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_mul, "AddMultiplyFusion");
    this->register_matcher(m, callback);
}

ngraph::pass::AddAddFusion::AddAddFusion() {
    // Create Add->Add pattern where first Add has exactly one consumer
    auto m_data = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto m_add1_constant = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Constant>());
    auto m_add1 = std::make_shared<pattern::op::Any>(element::f32, Shape{},
            [](std::shared_ptr<Node> node) {
                // Check that node has type opset3::Add and node has only one consumer
                return std::dynamic_pointer_cast<opset3::Add>(node) && node->output(0).get_target_inputs().size() == 1;
            }, NodeVector{m_data, m_add1_constant});
    auto m_add2_constant = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Constant>());
    auto m_add2 = std::make_shared<ngraph::opset3::Add>(m_add1, m_add2_constant);

    ngraph::graph_rewrite_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        auto add2 = m.get_match_root();

        auto & label_to_output = m.get_pattern_value_map();
        Output<Node> input = label_to_output[m_data];
        Output<Node> add1_const = label_to_output[m_add1_constant];
        Output<Node> add2_const = label_to_output[m_add2_constant];

        // Replace Add->Add with single Add
        // Add operation will be added to the list of ops requested for pattern matching
        auto new_add = register_new_node<opset3::Add>(input, eltwise_fold<opset3::Add>(add1_const, add2_const));

        copy_runtime_info(add2, new_add);
        new_add->set_friendly_name(add2->get_friendly_name());
        replace_node(add2, new_add);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_add2, "AddAddFusion");
    this->register_matcher(m, callback);
}

ngraph::pass::MultiplyMultiplyFusion::MultiplyMultiplyFusion() {
    // Create Multiply->Multiply pattern where first Multiply has exactly one consumer
    auto m_data = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto m_mul1_constant = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Constant>());
    auto m_mul1 = std::make_shared<pattern::op::Any>(element::f32, Shape{},
            [](std::shared_ptr<Node> node) {
                // Check that node has type opset3::Multiply and node has only one consumer
                return std::dynamic_pointer_cast<opset3::Multiply>(node) && node->output(0).get_target_inputs().size() == 1;
            }, NodeVector{m_data, m_mul1_constant});
    auto m_mul2_constant = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Constant>());
    auto m_mul2 = std::make_shared<ngraph::opset3::Multiply>(m_mul1, m_mul2_constant);

    ngraph::graph_rewrite_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        auto mul2 = m.get_match_root();

        auto & label_to_output = m.get_pattern_value_map();
        Output<Node> input = label_to_output[m_data];
        Output<Node> mul1_const = label_to_output[m_mul1_constant];
        Output<Node> mul2_const = label_to_output[m_mul2_constant];

        // Replace Multiply->Multiply with single Multiply
        // Multiply operation will be added to the list of ops requested for pattern matching
        auto new_mul = register_new_node<opset3::Multiply>(input, eltwise_fold<opset3::Multiply>(mul1_const, mul2_const));

        copy_runtime_info(mul2, new_mul);
        new_mul->set_friendly_name(mul2->get_friendly_name());
        replace_node(mul2, new_mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_mul2, "MultiplyMultiplyFusion");
    this->register_matcher(m, callback);
}
