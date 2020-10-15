//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/loop.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/specialize_function.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v5::Loop, "Loop", 5);

op::v5::Loop::Loop()
{
    // default trip_count, execution_condition
    auto trip_count =
        std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, -1);
    auto execution_condition =
        std::make_shared<ngraph::op::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);
    set_argument(0, Output<Node>(trip_count));
    set_argument(1, Output<Node>(execution_condition));
}

op::v5::Loop::Loop(const Output<Node>& trip_count,
                   const Output<Node>& execution_condition,
                   const OutputVector& args)
    : op::util::SubGraphOp({trip_count, execution_condition})
{
    set_arguments(args);
}

bool op::v5::Loop::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("body", m_body);
    visitor.on_attribute("input_descriptions", m_input_descriptions);
    visitor.on_attribute("output_descriptions", m_output_descriptions);

    return false;
}

void op::v5::Loop::validate_and_infer_types()
{
    if (m_special_body_ports.current_iteration_input_idx >= 0)
    {
        const auto& cur_iter_rank =
            m_body->get_parameters()[m_special_body_ports.current_iteration_input_idx]
                ->get_partial_shape()
                .rank();
        if (cur_iter_rank.is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  cur_iter_rank.compatible(1) || cur_iter_rank.compatible(0),
                                  "Rank of CurrentIteration input must be equal to 0 or 1");
        }
    }
    bool zero_number_of_iter = false;
    const auto& loop_execution_condition = input_value(1);
    const auto& loop_condition_rank = loop_execution_condition.get_partial_shape().rank();
    if (loop_condition_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              loop_condition_rank.compatible(1) ||
                                  loop_condition_rank.compatible(0),
                              "Rank of ExecutionCondition input must be equal to 0 or 1");
    }
    if (const auto& cond_value = std::dynamic_pointer_cast<const ngraph::opset5::Constant>(
            loop_execution_condition.get_node_shared_ptr()))
    {
        auto val = cond_value->cast_vector<bool>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the Condition constant is greater than 1");

        if (!val[0])
        {
            zero_number_of_iter = true;
        }
    }

    bool condition_always_true = false;
    NODE_VALIDATION_CHECK(this,
                          m_special_body_ports.body_condition_output_idx >= 0,
                          "Condition body output is not provided. "
                          "Condition is a mandatory output of the body in Loop op.");
    const auto& body_execution_condition =
        m_body->get_results()[m_special_body_ports.body_condition_output_idx]->input_value(0);
    const auto& body_condition_rank = body_execution_condition.get_partial_shape().rank();
    if (body_condition_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              body_condition_rank.compatible(0) ||
                                  body_condition_rank.compatible(1),
                              "Rank of BodyExecutionCondition output must be equal to 0 or 1");
    }
    if (const auto& cond_value = std::dynamic_pointer_cast<const ngraph::opset5::Constant>(
            body_execution_condition.get_node_shared_ptr()))
    {
        auto val = cond_value->cast_vector<bool>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the Condition constant is greater than 1");

        if (val[0])
        {
            condition_always_true = true;
        }
        else
        {
            m_num_iterations = 1; // condition_always_false, do_while mode
        }
    }

    const auto& trip_count = input_value(0);
    const auto& trip_count_rank = trip_count.get_partial_shape().rank();
    if (trip_count_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              trip_count_rank.compatible(1) || trip_count_rank.compatible(0),
                              "Rank of TripCount input must be equal to 0 or 1");
    }
    if (const auto& trip_count_val = std::dynamic_pointer_cast<const ngraph::opset5::Constant>(
            trip_count.get_node_shared_ptr()))
    {
        auto val = trip_count_val->cast_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              val.size() == 1,
                              "The number of values in the TripCount constant is greater than 1");
        if (condition_always_true)
            m_num_iterations = val[0];
    }

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions.size() + 2,
                          "Number of inputs must be the same as number of input descriptions");

    NODE_VALIDATION_CHECK(this,
                          get_output_size() == m_output_descriptions.size(),
                          "Number of outputs must be the same as number of output descriptions");

    std::vector<std::shared_ptr<Node>> ends;

    // Input
    uint64_t index_it = 2;
    for (const auto& input_description : m_input_descriptions)
    {
        auto index = input_description->m_input_index;
        NODE_VALIDATION_CHECK(this, index == index_it, "Input_index not in order");
        index_it++;

        if (auto merged_input_description = as_type_ptr<MergedInputDescription>(input_description))
        {
            auto body_value =
                m_body->get_results().at(merged_input_description->m_body_value_index)->input(0);
            ends.push_back(body_value.get_node()->shared_from_this());

            const auto& body_value_partial_shape = body_value.get_partial_shape();
            auto body_parameter =
                m_body->get_parameters().at(merged_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            NODE_VALIDATION_CHECK(this,
                                  body_value_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator successive value is not compatible with body param");
            NODE_VALIDATION_CHECK(this,
                                  input_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator initial value is not compatible with body param");

            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                // infer type for body_parameter
                if (body_param_partial_shape.is_dynamic())
                {
                    body_parameter->set_partial_shape(input_shape);
                }
            }
        }
        else if (auto invariant_input_description =
                     as_type_ptr<TensorIterator::InvariantInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(invariant_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            NODE_VALIDATION_CHECK(this,
                                  input_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator initial value is not compatible with body param");

            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                // infer type for m_body_parameter
                if (body_param_partial_shape.is_dynamic())
                {
                    body_parameter->set_partial_shape(input_shape);
                }
            }
        }
    }

    // Body
    m_body->validate_nodes_and_infer_types();

    // Output
    index_it = 0;
    for (const auto& output_description : m_output_descriptions)
    {
        auto index = output_description->m_output_index;
        NODE_VALIDATION_CHECK(this, index == index_it, "Output_index not in order");
        index_it++;

        auto body_value =
            m_body->get_results().at(output_description->m_body_value_index)->input_value(0);

        if (auto concat_output_description =
                as_type_ptr<TensorIterator::ConcatOutputDescription>(output_description))
        {
            const auto& body_value_partial_shape = body_value.get_partial_shape();
            set_output_type(index, body_value.get_element_type(), PartialShape::dynamic());
            if (body_value_partial_shape.is_static())
            {
                auto body_value_shape = body_value_partial_shape.to_shape();
                auto axis = concat_output_description->m_axis;

                Shape out_shape{body_value_shape};

                if (body_value_shape.empty())
                {
                    NODE_VALIDATION_CHECK(
                        this,
                        axis == 0,
                        "Axis must be equal to 0 if concatenated output tensor slices are scalars. "
                        "Loop output index: ",
                        index);
                    out_shape = Shape(1);
                }

                if (m_num_iterations != -1)
                {
                    out_shape[axis] = m_num_iterations * body_value_shape[axis];
                    if (zero_number_of_iter)
                    {
                        out_shape.at(0) = 0;
                    }
                    set_output_type(index, body_value.get_element_type(), out_shape);
                }
            }
        }
        else if (auto body_output_description =
                     as_type_ptr<TensorIterator::BodyOutputDescription>(output_description))
        {
            const PartialShape& ps = body_value.get_partial_shape();
            if (ps.is_dynamic())
            {
                set_output_type(index, body_value.get_element_type(), ps);
            }
            else
            {
                auto shape = ps.get_shape();
                if (zero_number_of_iter)
                {
                    shape.at(0) = 0;
                }
                set_output_type(index, body_value.get_element_type(), shape);
            }
        }
    }
}

std::shared_ptr<Node> op::v5::Loop::clone_with_new_inputs(const OutputVector& new_args) const
{
    // 0 - trip_count, 1 - execution condition, these inputs are not connected to the body params
    OutputVector body_params_args(new_args.begin() + 2, new_args.end());
    auto op = make_shared<op::v5::Loop>(new_args[0], new_args[1], body_params_args);
    NGRAPH_CHECK(op.get(),
                 op != nullptr,
                 "Cannot clone ",
                 description(),
                 " operation with name ",
                 get_friendly_name());
    op->set_output_size(m_output_descriptions.size());

    std::vector<::ngraph::element::Type> types(m_body->get_parameters().size());
    std::vector<::ngraph::PartialShape> new_shapes(m_body->get_parameters().size());

    for (size_t input_index = 0; input_index < new_args.size(); ++input_index)
    {
        for (auto& input_description : m_input_descriptions)
        {
            if (input_description->m_input_index == input_index)
            {
                types[input_description->m_body_parameter_index] =
                    new_args[input_index].get_element_type();
                new_shapes[input_description->m_body_parameter_index] =
                    new_args[input_index].get_partial_shape();
            }
        }
    }

    if (m_special_body_ports.current_iteration_input_idx >= 0)
    {
        const auto& cur_iterations_param =
            m_body->get_parameters()[m_special_body_ports.current_iteration_input_idx];
        body_params_args.insert(
            body_params_args.begin() + m_special_body_ports.current_iteration_input_idx,
            m_body->get_parameters()[m_special_body_ports.current_iteration_input_idx]);
        new_shapes[m_special_body_ports.current_iteration_input_idx] =
            cur_iterations_param->get_partial_shape();
        types[m_special_body_ports.current_iteration_input_idx] =
            cur_iterations_param->get_element_type();
    }
    op->m_num_iterations = m_num_iterations;
    op->m_special_body_ports = m_special_body_ports;
    auto func = std::make_shared<Function>(m_body->get_results(), m_body->get_parameters());
    auto spec_func = specialize_function(
        func, types, new_shapes, std::vector<void*>(body_params_args.size(), nullptr));
    op->m_body = std::make_shared<Function>(spec_func->get_results(), spec_func->get_parameters());

    for (auto& input_description : m_input_descriptions)
    {
        op->m_input_descriptions.push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions)
    {
        op->m_output_descriptions.push_back(output_description->copy());
    }
    return move(op);
}

Output<Node> op::v5::Loop::get_concatenated_slices(const Output<Node>& value,
                                                   int64_t start,
                                                   int64_t stride,
                                                   int64_t part_size,
                                                   int64_t end,
                                                   int64_t axis)
{
    NGRAPH_CHECK(start == 0 && stride == 1 && part_size == 1 && end == -1,
                 "Invalid start, stride, part_size, or end attribute values in Loop op. "
                 "Supported values for start {0}, for stride and part_size {1}, for end "
                 "{-1}");
    return SubGraphOp::get_concatenated_slices(value, start, stride, part_size, end, axis);
}

void op::v5::Loop::set_sliced_input(const shared_ptr<Parameter>& parameter,
                                    const Output<Node>& value,
                                    int64_t start,
                                    int64_t stride,
                                    int64_t part_size,
                                    int64_t end,
                                    int64_t axis)
{
    NGRAPH_CHECK(false,
                 "Incorrect type of input. Implicit slicing is not supported in "
                 "Loop operation.");
}
