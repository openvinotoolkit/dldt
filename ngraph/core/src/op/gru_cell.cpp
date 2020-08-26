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

#include <cmath>
#include <functional>

#include "itt.hpp"
#include "ngraph/runtime/reference/gru_cell.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/gru_cell.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::v3::GRUCell::type_info;

op::v3::GRUCell::GRUCell()
    : m_linear_before_reset(false)
{
    m_activations = {"sigmoid", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size)
    : GRUCell(X,
              initial_hidden_state,
              W,
              R,
              hidden_size,
              vector<string>{"sigmoid", "tanh"},
              vector<float>{},
              vector<float>{},
              0.f,
              false)
{
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size,
                         const vector<string>& activations,
                         const vector<float>& activations_alpha,
                         const vector<float>& activations_beta,
                         float clip,
                         bool linear_before_reset)
    : Op({X, initial_hidden_state, W, R})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_linear_before_reset{linear_before_reset}
{
    add_default_bias_input();
    constructor_validate_and_infer_types();
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         size_t hidden_size,
                         const vector<string>& activations,
                         const vector<float>& activations_alpha,
                         const vector<float>& activations_beta,
                         float clip,
                         bool linear_before_reset)
    : Op({X, initial_hidden_state, W, R, B})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_linear_before_reset{linear_before_reset}
{
    constructor_validate_and_infer_types();
}

bool op::v3::GRUCell::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("linear_before_reset", m_linear_before_reset);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

void op::v3::GRUCell::validate_and_infer_types()
{
    std::vector<ngraph::PartialShape> input_param{};

    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto result_et = element::dynamic;

    // Copy all inputs for further validation
    for (size_t i = 0; i < get_input_size(); i++)
    {
        input_param.push_back(get_input_partial_shape(i));
    }

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& w_pshape = get_input_partial_shape(2);
    const auto& r_pshape = get_input_partial_shape(3);
    const auto& b_pshape = get_input_partial_shape(4);

    validate_input_rank_dimension(input_param);

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(4)),
        "Element types for X, initial_hidden_state, W, R and B do not match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]),
                          "Parameter batch_size not matched for ht_pshape and x_pshape.");

    // Merge hidden_size dimension across all inputs to evaluate output[1] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[1]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[1]),
                          "Parameter hidden_size not matched for ht_pshape and t_pshape.");

    // Validate hidden_size value for W, B and R inputs
    if (merged_hidden_size.is_static())
    {
        if (w_pshape[0].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                w_pshape[0].compatible(merged_hidden_size * s_gates_count),
                "Parameter hidden_size mistmatched in w_pshape. Current value is: ",
                w_pshape[0].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * s_gates_count,
                ".");
        }

        if (r_pshape[0].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                r_pshape[0].compatible(merged_hidden_size * s_gates_count),
                "Parameter hidden_size mistmatched in r_pshape. Current value is: ",
                r_pshape[0].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * s_gates_count,
                ".");
        }

        if (b_pshape[0].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                b_pshape[0].compatible(merged_hidden_size *
                                       (s_gates_count + m_linear_before_reset)),
                "Parameter hidden_size mistmatched in b_pshape. Current value is: ",
                b_pshape[0].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * (s_gates_count + m_linear_before_reset),
                ".");
        }
    }

    // Mark inputs which are relevant to output parameters
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(3);

    // Set output size, type and shape
    set_output_size(1);
    set_output_type(0, result_et, {merged_batch_size, merged_hidden_size});
}

void op::v3::GRUCell::add_default_bias_input()
{
    Output<Node> B = op::Constant::create(
        get_input_element_type(0),
        Shape{(s_gates_count + m_linear_before_reset) * get_hidden_size()},
        vector<float>((s_gates_count + m_linear_before_reset) * get_hidden_size(), 0.f));
    set_argument(4, B);
}

shared_ptr<Node> op::v3::GRUCell::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 4)
    {
        return make_shared<GRUCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    get_hidden_size(),
                                    get_activations(),
                                    get_activations_alpha(),
                                    get_activations_beta(),
                                    get_clip(),
                                    m_linear_before_reset);
    }
    else if (new_args.size() == 5)
    {
        return make_shared<GRUCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    new_args.at(4),
                                    get_hidden_size(),
                                    get_activations(),
                                    get_activations_alpha(),
                                    get_activations_beta(),
                                    get_clip(),
                                    m_linear_before_reset);
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg1,
                  const HostTensorPtr& arg2,
                  const HostTensorPtr& arg3,
                  const HostTensorPtr& arg4,
                  const HostTensorPtr& arg5,
                  const HostTensorPtr& out,
                  const std::string& activation_f,
                  const std::string& activation_g,
                  float clip,
                  bool linear_before_reset)
    {
        runtime::reference::gru_cell(arg1->get_data_ptr<ET>(),
                                     arg1->get_shape(),
                                     arg2->get_data_ptr<ET>(),
                                     arg2->get_shape(),
                                     arg3->get_data_ptr<ET>(),
                                     arg3->get_shape(),
                                     arg4->get_data_ptr<ET>(),
                                     arg4->get_shape(),
                                     arg5->get_data_ptr<ET>(),
                                     arg5->get_shape(),
                                     out->get_data_ptr<ET>(),
                                     activation_f,
                                     activation_g,
                                     clip,
                                     linear_before_reset);
        return true;
    }

    bool evaluate_gru_cell(const HostTensorPtr& arg1,
                           const HostTensorPtr& arg2,
                           const HostTensorPtr& arg3,
                           const HostTensorPtr& arg4,
                           const HostTensorPtr& arg5,
                           const HostTensorPtr& out,
                           const std::string& activation_f,
                           const std::string& activation_g,
                           float clip,
                           bool linear_before_reset)
    {
        element::Type_t axis_type = arg2->get_element_type();
        bool rc = true;
        switch (axis_type)
        {
            TYPE_CASE(f32)(arg1, arg2, arg3, arg4, arg5, out, activation_f, activation_g, clip, linear_before_reset);
                break;
                // todo: determinate necessary types
            default: rc = false; break;
        }
        return rc;
    }
}

bool op::GRUCell::evaluate(const HostTensorVector& output_values,
                           const HostTensorVector& input_values) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::RNNCell::evaluate");
    return evaluate_gru_cell(input_values[0], input_values[1], input_values[2],
                             input_values[3], input_values[4], output_values[0], m_activations[0], m_activations[1],
                             m_clip, m_linear_before_reset);
}
