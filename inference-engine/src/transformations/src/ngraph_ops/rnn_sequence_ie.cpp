// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/rnn_sequence_ie.hpp"
#include "ngraph/op/util/recurrent_sequence.hpp"
#include "itt.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::RNNSequenceIE, "RNNSequenceIE", 4);

op::RNNSequenceIE::RNNSequenceIE(const Output<Node>& X,
                                 const Output<Node>& H_t,
                                 const Output<Node>& seq_lengths, // actually not supported
                                 const Output<Node>& WR,
                                 const Output<Node>& B,
                                 std::size_t hidden_size,
                                 op::RecurrentSequenceDirection direction,
                                 const std::vector<std::string>& activations,
                                 const std::vector<float>& activations_alpha,
                                 const std::vector<float>& activations_beta,
                                 float clip)
        : RNNCellBase({X, H_t, seq_lengths, WR, B}, hidden_size, clip, activations, activations_alpha, activations_beta),
          m_direction(direction) {
    constructor_validate_and_infer_types();
}

void op::RNNSequenceIE::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(RNNSequenceIE_validate_and_infer_types,
        for (const auto& input : inputs()) {
            if (input.get_partial_shape().rank().is_dynamic()) {
                set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
                set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
                return;
            }
        }
        // rank validation
        auto x_pshape = get_input_partial_shape(0);
        auto h_state_pshape = get_input_partial_shape(1);
        auto seq_lengths_pshape = get_input_partial_shape(2);
        auto wr_pshape = get_input_partial_shape(3);
        auto b_pshape = get_input_partial_shape(4);

        std::vector<ngraph::PartialShape> pshapes = {x_pshape, h_state_pshape, seq_lengths_pshape, wr_pshape, b_pshape};
        std::vector<std::string> in_names = {"X", "H", "seq_lenghts", "WR", "B"};
        // num_direction dimension should be squeezed, we don't support bidirectional case
        std::vector<size_t> ranks = {3, 2, 1, 2, 1};
        for (size_t i = 0; i < pshapes.size(); ++i) {
            NGRAPH_CHECK((pshapes[i].rank().get_length() == ranks[i]),
                        "RNNSequenceIE ",
                        in_names[i],
                        " input rank is not correct.");
        }

        element::Type arg_type = get_input_element_type(0);
        PartialShape output_shape_0{PartialShape::dynamic(3)};
        PartialShape output_shape_1{PartialShape::dynamic(2)};
        if (get_input_partial_shape(0).is_static()) {
            size_t batch_size = get_input_partial_shape(0).get_shape()[0];
            size_t seq_length = get_input_partial_shape(0).get_shape()[1];
            output_shape_0 = Shape{batch_size, seq_length, m_hidden_size};
            output_shape_1 = Shape{batch_size, m_hidden_size};
        }
        set_output_type(0, arg_type, output_shape_0);
        set_output_type(1, arg_type, output_shape_1);
        return;
    )
    NODE_VALIDATION_CHECK(this, false, "Function is not included into the selective build.");
}

bool op::RNNSequenceIE::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(RNNSequenceIE_visit_attributes,
        visitor.on_attribute("direction", m_direction);
        return op::util::RNNCellBase::visit_attributes(visitor);
    )
    return false;
}

shared_ptr<Node> op::RNNSequenceIE::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::RNNSequenceIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
             new_args.at(4), m_hidden_size, m_direction, m_activations, m_activations_alpha, m_activations_beta, m_clip);
}
