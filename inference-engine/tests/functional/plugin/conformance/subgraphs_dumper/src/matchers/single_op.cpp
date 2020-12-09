// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/single_op.hpp"
#include "ngraph/ops.hpp"

using namespace SubgraphsDumper;

// TODO: Move to some utils?
bool compare_constants_data(const std::shared_ptr<ngraph::op::Constant> &op,
                            const std::shared_ptr<ngraph::op::Constant> &ref) {
    switch (op->get_element_type()) {
        case ngraph::element::Type_t::boolean:
            if (op->cast_vector<bool>() != ref->cast_vector<bool>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::bf16:
            if (op->cast_vector<ngraph::bfloat16>() != ref->cast_vector<ngraph::bfloat16>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::f16:
            if (op->cast_vector<ngraph::float16>() != ref->cast_vector<ngraph::float16>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::f32:
            if (op->cast_vector<float>() != ref->cast_vector<float>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::f64:
            if (op->cast_vector<double>() != ref->cast_vector<double>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i8:
            if (op->cast_vector<int8_t>() != ref->cast_vector<int8_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i16:
            if (op->cast_vector<int16_t>() != ref->cast_vector<int16_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i32:
            if (op->cast_vector<int32_t>() != ref->cast_vector<int32_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i64:
            if (op->cast_vector<int64_t>() != ref->cast_vector<int64_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u8:
            if (op->cast_vector<uint8_t>() != ref->cast_vector<uint8_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u16:
            if (op->cast_vector<uint16_t>() != ref->cast_vector<uint16_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u32:
            if (op->cast_vector<uint32_t>() != ref->cast_vector<uint32_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u64:
            if (op->cast_vector<uint64_t>() != ref->cast_vector<uint64_t>()) {
                return false;
            } else {
                return true;
            }
        default:
            throw std::runtime_error("unsupported type");
    }
}

const char *SingleOpMatcher::name = "generic_single_op";

bool SingleOpMatcher::match(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) {
    // Match OP type and version
    if (node->get_type_info().name != ref->get_type_info().name ||
        node->get_type_info().version != ref->get_type_info().version) {
        return false;
    }
    // Match inputs size
    if (node->get_input_size() == ref->get_input_size()) {
        // Match input ranks, element types and static shapes
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            bool rankIsEqual = node->get_input_tensor(i).get_partial_shape().rank() ==
                               ref->get_input_tensor(i).get_partial_shape().rank();
            bool elemTypeIsEqual = node->get_input_tensor(i).get_element_type() ==
                                   ref->get_input_tensor(i).get_element_type();
            bool is_dynamic = node->get_input_node_ptr(i)->is_dynamic() ==
                              ref->get_input_node_ptr(i)->is_dynamic();
            if (!(rankIsEqual && elemTypeIsEqual && is_dynamic)) {
                return false;
            }
        }
    } else {
        return false;
    }

    // Match outputs size
    if (node->get_output_size() == ref->get_output_size()) {
        // Match output element type
        for (size_t i = 0; i < node->get_output_size(); ++i) {
            if (node->get_output_tensor(i).get_element_type() !=
                ref->get_output_tensor(i).get_element_type()) {
                return false;
            }
        }
    } else {
        return false;
    }
    // TODO: Figure Out with visitors
//    ngraph::AttributeVisitor visitor();
//    node->visit_attributes(visitor);
    // Match ports values
    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        if (std::find(ignore_const_port_vals.begin(), ignore_const_port_vals.end(), port_id) !=
            ignore_const_port_vals.end()) {
            continue;
        }
        const auto &cur_node_input = node->input_value(port_id).get_node_shared_ptr();
        const auto &ref_node_input = ref->input_value(port_id).get_node_shared_ptr();

        const auto &cur_const_input = std::dynamic_pointer_cast<ngraph::op::Constant>(cur_node_input);
        const auto &ref_const_input = std::dynamic_pointer_cast<ngraph::op::Constant>(cur_node_input);

        // Check that both OP an reference port inputs are constant and have same data
        if (cur_const_input != nullptr && ref_const_input != nullptr &&
            !compare_constants_data(cur_const_input, ref_const_input)) {
            return false;
            // Check that input nodes on the port both not constants
        } else if ((cur_const_input != nullptr && ref_const_input == nullptr) ||
                   (cur_const_input == nullptr && ref_const_input != nullptr)) {
            return false;
        }
    }
    return true;
}
