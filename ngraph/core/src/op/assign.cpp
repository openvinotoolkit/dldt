//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/assign.hpp"
#include "itt.hpp"
#include "ngraph/op/read_value.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/ops.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::AssignBase, "AssignBase", 0);
NGRAPH_RTTI_DEFINITION(op::v3::Assign, "Assign", 3, op::Sink);
NGRAPH_RTTI_DEFINITION(op::v6::Assign, "Assign", 6, op::Sink);

op::v3::Assign::Assign(const Output<Node>& new_value, const std::string& variable_id)
    : AssignBase({new_value})
    , m_variable_id(variable_id)
{
    constructor_validate_and_infer_types();
}

void op::v3::Assign::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_Assign_validate_and_infer_types);
    auto value = input_value(0);
    auto arg_t = get_input_element_type(0);
    auto output_shape = get_input_partial_shape(0);
    if (!m_variable)
    {
        NodeVector start_nodes;
        for (const auto& input : inputs())
        {
            start_nodes.push_back(input.get_source_output().get_node_shared_ptr());
        }
        auto nodes = topological_sort(start_nodes);
        for (const auto& node : nodes)
        {
            if (auto read_value = as_type_ptr<op::v3::ReadValue>(node))
            {
                if (read_value->get_variable_id() == m_variable_id)
                    m_variable = read_value->get_variable();
            }
        }
        NODE_VALIDATION_CHECK(
            this, m_variable != nullptr, "Can't find variable with id = ", m_variable_id);
    }

    auto variable_info = m_variable->get_info();
    NODE_VALIDATION_CHECK(this,
                          m_variable_id == variable_info.variable_id,
                          "Variables identifiers are inconsistent.");
    NODE_VALIDATION_CHECK(
        this, arg_t == variable_info.data_type, "Variables types are inconsistent.");

    if (output_shape.is_static() && variable_info.data_shape.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              output_shape == variable_info.data_shape,
                              "Variables output shapes are inconsistent.");

        set_output_type(0, arg_t, output_shape);
    }
    else
    {
        set_output_type(0, arg_t, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::v3::Assign::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_Assign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v3::Assign>(new_args.at(0), m_variable_id);
}

bool op::v3::Assign::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_Assign_visit_attributes);
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}

op::v6::Assign::Assign(const Output<Node>& new_value, const std::shared_ptr<Variable>& variable)
    : AssignBase({new_value})
{
    m_variable = variable;
    constructor_validate_and_infer_types();
}

void op::v6::Assign::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_Assign_validate_and_infer_types);
    m_variable->update({get_input_partial_shape(0),
                        get_input_element_type(0),
                        m_variable->get_info().variable_id});
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v6::Assign::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_Assign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v6::Assign>(new_args.at(0), m_variable);
}

bool op::v6::Assign::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_Assign_visit_attributes);
    visitor.on_attribute("variable_id", m_variable);
    return true;
}

bool op::v6::Assign::evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs,
                              const EvaluationContext& evaluation_context) const
{
    NGRAPH_OP_SCOPE(v6_Assign_evaluate);
    const auto& variable_context = evaluation_context.get_variable_context();
    const auto& var_value = variable_context.find(m_variable);

    // todo: exception?
    NODE_VALIDATION_CHECK(this,
                          var_value != variable_context.end(),
                          "No context found for ",
                          m_variable->get_info().variable_id,
                          " variable.");

    var_value->second->set_reset(false);
    const auto& value = var_value->second->get_value();
    value->set_unary(inputs[0]);
    outputs[0]->set_unary(inputs[0]);

    void *output = outputs[0]->get_data_ptr();
    void *input = inputs[0]->get_data_ptr();
    memcpy(output, input, outputs[0]->get_size_in_bytes());
    memcpy(value->get_data_ptr(), input, value->get_size_in_bytes());
    return true;
}

bool op::v6::Assign::constant_fold(OutputVector& output_values, const OutputVector& inputs_values)
{
    return false;
}