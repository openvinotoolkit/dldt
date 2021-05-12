// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "unsqueeze.hpp"
#include <ngraph/opsets/opset6.hpp>
#include <paddlepaddle_frontend/utility.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

namespace pdpd
{
    namespace op
    {
        NamedOutputs unsqueeze(const NodeContext& node)
        {
            // TODO to support data type other than int32_t #55168
            auto data = node.get_ng_input("X");
            auto axes = node.get_attribute<std::vector<int32_t>>("axes");
            auto axesNode = opset6::Constant::create(element::i32, {axes.size()}, axes);
            return node.default_single_output_mapping(
                {std::make_shared<opset6::Unsqueeze>(data, axesNode)}, {"Out"});
        }

    } // namespace op
} // namespace pdpd
