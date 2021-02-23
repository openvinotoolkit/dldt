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

#include <numeric>

#include "default_opset.hpp"
#include "ngraph/shape.hpp"
#include "op/add.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector add(const Node& node)
                {
                    const Output<ngraph::Node> lhs_node = node.get_ng_inputs().at(0);
                    Output<ngraph::Node> rhs_node = node.get_ng_inputs().at(1);
                    bool broadcast = node.get_attribute_value<std::int64_t>("broadcast", 0);
                    if (broadcast)
                    {
                        if (node.has_attribute("axis"))
                        {
                            // Unidirectional broadcast right node to left shape.
                            const auto rhs_rank = rhs_node.get_partial_shape().rank();
                            NGRAPH_CHECK(rhs_rank.is_static(),
                                         "Add operator second tensor's rank has to be static.");
                            const auto axis = node.get_attribute_value<std::int64_t>("axis");
                            const auto axes_num = static_cast<size_t>(rhs_rank.get_length());
                            std::vector<int64_t> axes_vals(axes_num);
                            std::iota(axes_vals.begin(), axes_vals.end(), axis);

                            rhs_node = std::make_shared<default_opset::Broadcast>(
                                rhs_node,
                                std::make_shared<default_opset::ShapeOf>(lhs_node),
                                default_opset::Constant::create(
                                    element::i64, Shape{axes_num}, axes_vals));
                        }
                        else
                        {
                            rhs_node = std::make_shared<default_opset::Broadcast>(
                                rhs_node, std::make_shared<default_opset::ShapeOf>(lhs_node));
                        }
                        return {std::make_shared<default_opset::Add>(
                            lhs_node, rhs_node, ngraph::op::AutoBroadcastSpec::NONE)};
                    }
                    return {std::make_shared<default_opset::Add>(lhs_node, rhs_node)};
                }

            } // namespace set_1

            namespace set_7
            {
                OutputVector add(const Node& node)
                {
                    return {std::make_shared<default_opset::Add>(node.get_ng_inputs().at(0),
                                                                 node.get_ng_inputs().at(1))};
                }

            } // namespace set_7

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
