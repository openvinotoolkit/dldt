// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_precision.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset1.hpp>

using namespace ngraph;

bool fuse_type_to_constant(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, const std::vector<ngraph::Input<ngraph::Node>> & consumers);
bool fuse_type_to_shapeof(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_parameter(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_convert(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_nms3(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_nms4(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_topk(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_nonzero(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);
bool fuse_type_to_bucketize(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx);

static std::map<ngraph::NodeTypeInfo, std::function<bool(std::shared_ptr<Node>&, element::Type, size_t idx)>> type_to_fuse {
        {opset4::Parameter::type_info, fuse_type_to_parameter},
        {opset4::Convert::type_info, fuse_type_to_convert},
        {opset4::ShapeOf::type_info, fuse_type_to_shapeof},
        {opset3::NonMaxSuppression::type_info, fuse_type_to_nms3},
        {opset4::NonMaxSuppression::type_info, fuse_type_to_nms4},
        {opset4::TopK::type_info, fuse_type_to_topk},
        {opset4::NonZero::type_info, fuse_type_to_nonzero},
        {opset4::Bucketize::type_info, fuse_type_to_bucketize},
};

bool ngraph::pass::ConvertPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // As Constant operations can be shared between multiple nGraph Functions so before
    // changing precision we need to understand which Constant consumers belongs
    // to the current nGraph Function
    std::map<std::shared_ptr<Node>, std::vector<Input<Node>>> const_to_internal_output;

    std::function<void(const std::shared_ptr<Function> &)> register_constants =
            [&const_to_internal_output](const std::shared_ptr<Function> & f) {
        for (auto & node : f->get_ordered_ops()) {
            for (auto & input : node->inputs()) {
                if (auto const_node = std::dynamic_pointer_cast<opset4::Constant>(input.get_source_output().get_node_shared_ptr())) {
                    const_to_internal_output[const_node].emplace_back(input);
                }
            }
        }
    };

    register_constants(f);

    auto convert_node_precision = [this, &const_to_internal_output](std::shared_ptr<Node> & node) {
        // As input type could changed we need to propagate output type calculation manually
        node->validate_and_infer_types();

        for (auto output : node->outputs()) {
            if (output.get_element_type() == m_from) {
                // Handle case with Constants as they can have consumers from other nGraph Function object
                if (ngraph::op::is_constant(node) && const_to_internal_output.count(node)) {
                    fuse_type_to_constant(node, m_to, const_to_internal_output.at(node));
                    continue;
                }

                // If node type in map and convert can be fused into node we skip Convert creation
                if (type_to_fuse.count(node->get_type_info()) &&
                    type_to_fuse.at(node->get_type_info())(node, m_to, output.get_index())) {
                    node->validate_and_infer_types();
                    continue;
                }

                // Create Convert operation and reconnect consumers
                auto consumers = output.get_target_inputs();
                auto convert = std::make_shared<opset4::Convert>(output, m_to);
                for (auto & input : consumers) {
                    input.replace_source_output(convert);
                }
            }
        }
    };

    std::function<void(const std::shared_ptr<Function> &)> convert_function_precision =
            [this, &const_to_internal_output, &convert_node_precision](const std::shared_ptr<Function> & f) {
        // Iterate over all nodes in topological order and then iterate over node outputs.
        // If output type mismatch given type we try to fuse type into this operation
        // otherwise we insert Convert operation.
        for (auto &node : f->get_ordered_ops()) {
            convert_node_precision(node);
        }
    };

    convert_function_precision(f);

    // TODO: we need to split NopElimination pass to separate MatcherPasses and call Convert elimination here
    for (auto &node : f->get_ordered_ops()) {
        if (auto convert = std::dynamic_pointer_cast<opset4::Convert>(node)) {
            if (convert->input(0).get_element_type() == convert->get_convert_element_type()) {
                replace_output_update_name(convert->output(0), convert->input_value(0));
            }
        }
    }
    return true;
}

bool fuse_type_to_shapeof(std::shared_ptr<Node> & node, element::Type to, size_t idx) {
    if (auto shapeof = as_type_ptr<opset4::ShapeOf>(node)) {
        if (to == element::i32 || to == element::i64) {
            shapeof->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_parameter(std::shared_ptr<Node> & node, element::Type to, size_t idx) {
    if (auto param = as_type_ptr<opset4::Parameter>(node)) {
        param->set_element_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_convert(std::shared_ptr<Node> & node, element::Type to, size_t idx) {
    if (auto convert = as_type_ptr<opset4::Convert>(node)) {
        convert->set_convert_element_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms3(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto nms = as_type_ptr<opset3::NonMaxSuppression>(node)) {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms4(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto nms = as_type_ptr<opset4::NonMaxSuppression>(node)) {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_topk(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto topk = as_type_ptr<opset4::TopK>(node)) {
        if (idx == 1 && (to == element::i32 || to == element::i64)) {
            topk->set_index_element_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_nonzero(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto nonzero = as_type_ptr<opset4::NonZero>(node)) {
        if (to == element::i32 || to == element::i64) {
            nonzero->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_bucketize(std::shared_ptr<ngraph::Node> & node, ngraph::element::Type to, size_t idx) {
    if (auto b = as_type_ptr<opset4::Bucketize>(node)) {
        if (to == element::i32 || to == element::i64) {
            b->set_output_type(to);
            return true;
        }
    }
    return false;
}

template <element::Type_t PREC_FROM, element::Type_t PREC_TO>
std::shared_ptr<Node> change_constant_precision(std::shared_ptr<opset4::Constant> & constant) {
    using src_type = typename element_type_traits<PREC_FROM>::value_type;
    using dst_type = typename element_type_traits<PREC_TO>::value_type;

    std::vector<src_type> data(std::move(constant->get_vector<src_type>()));
    std::vector<dst_type> final_data;
    std::transform(data.begin(), data.end(), std::back_inserter(final_data),
                   [](src_type val) {
                       if (val > std::numeric_limits<dst_type>::max()) {
                           return std::numeric_limits<dst_type>::max();
                       } else {
                           return static_cast<dst_type>(val);
                       }
                   });
    return std::make_shared<ngraph::opset4::Constant>(PREC_TO, constant->get_shape(), final_data);
}

bool fuse_type_to_constant(std::shared_ptr<Node> & node, element::Type to, const std::vector<Input<Node>> & consumers) {
    if (auto constant = as_type_ptr<opset4::Constant>(node)) {
        auto from = constant->get_element_type();
        std::shared_ptr<Node> new_const;
        if (from == element::u64 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u64, element::Type_t::i32>(constant);
        } else if (from == element::i64 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::i64, element::Type_t::i32>(constant);
        } else if (from == element::u8 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u8, element::Type_t::i32>(constant);
        } else if (from == element::u16 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u16, element::Type_t::i32>(constant);
        } else if (from == element::u32 && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::u32, element::Type_t::i32>(constant);
        } else if (from == element::f16 && to == element::f32) {
            new_const = change_constant_precision<element::Type_t::f16, element::Type_t::f32>(constant);
        } else if (from == element::boolean && to == element::u8) {
            new_const = change_constant_precision<element::Type_t::boolean, element::Type_t::u8>(constant);
        } else if (from == element::boolean && to == element::i32) {
            new_const = change_constant_precision<element::Type_t::boolean, element::Type_t::i32>(constant);
        } else {
            throw ngraph_error("not supported");
        }
        for (auto & output : consumers) {
            output.replace_source_output(new_const);
        }

        new_const->validate_and_infer_types();
        if (constant->get_output_target_inputs(0).size() == consumers.size()) {
            new_const->set_friendly_name(constant->get_friendly_name());
        }
    }
    return false;
}
