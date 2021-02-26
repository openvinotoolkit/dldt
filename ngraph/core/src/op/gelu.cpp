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

#include <cmath>
#include "itt.hpp"

#include "ngraph/op/gelu.hpp"

#include <ngraph/validation_util.hpp>
#include "ngraph/runtime/reference/gelu.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::Gelu::type_info;

op::v0::Gelu::Gelu()
    : UnaryElementwiseArithmetic()
{
}

op::v0::Gelu::Gelu(const Output<Node>& data)
    : UnaryElementwiseArithmetic(data)
{
    constructor_validate_and_infer_types();
}

bool op::v0::Gelu::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Gelu_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::Gelu::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Gelu_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<op::v0::Gelu>(new_args.at(0));
}

void op::v0::Gelu::validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    set_output_type(0, input_element_type, input_pshape);
}

// ------------------------------ V6 ------------------------------

namespace ngraph
{
    template <>
    NGRAPH_API EnumNames<op::GeluApproximationMode>& EnumNames<op::GeluApproximationMode>::get()
    {
        static auto enum_names = EnumNames<op::GeluApproximationMode>(
            "op::GeluApproximationMode",
            {{"TANH", op::GeluApproximationMode::TANH}, {"ERF", op::GeluApproximationMode::ERF}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::GeluApproximationMode>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::GeluApproximationMode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

NGRAPH_RTTI_DEFINITION(op::v6::Gelu, "Gelu", 6);

op::v6::Gelu::Gelu(const Output<Node>& data, GeluApproximationMode mode)
    : UnaryElementwiseArithmetic(data)
    , m_approximation_mode(mode)
{
    constructor_validate_and_infer_types();
}

bool op::v6::Gelu::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_Gelu_visit_attributes);
    visitor.on_attribute("approximation_mode", m_approximation_mode);
    return true;
}

shared_ptr<Node> op::v6::Gelu::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_Gelu_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<op::v6::Gelu>(new_args.at(0), m_approximation_mode);
}

void op::v6::Gelu::validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    set_output_type(0, input_element_type, input_pshape);
}

op::GeluApproximationMode op::v6::Gelu::get_approximation_mode()
{
    return m_approximation_mode;
}

namespace gelu
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0,
                         const HostTensorPtr& out,
                         op::GeluApproximationMode mode,
                         const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gelu<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), mode, count);
        return true;
    }

    bool evaluate_gelu(const HostTensorPtr& arg0,
                       const HostTensorPtr& out,
                       op::GeluApproximationMode mode,
                       const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_gelu, f16, arg0, out, mode, count);
            NGRAPH_TYPE_CASE(evaluate_gelu, f32, arg0, out, mode, count);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v6::Gelu::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v6_Gelu_evaluate);
    return gelu::evaluate_gelu(
        inputs[0], outputs[0], m_approximation_mode, shape_size(get_output_shape(0)));
}
