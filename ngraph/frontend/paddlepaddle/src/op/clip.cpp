// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clip.hpp"
#include <paddlepaddle_frontend/exceptions.hpp>
#include <ngraph/opsets/opset6.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

namespace pdpd
{
    namespace op
    {
        NamedOutputs clip(const NodeContext& node)
        {
            auto data = node.get_ng_input("X");
            auto min = node.get_attribute<float>("min");
            auto max = node.get_attribute<float>("max");
            PDPD_NODE_VALIDATION_CHECK(ngraph::frontend::ErrorCode::OP_VALIDATION_FAILED, node, max >= min, "clip: max value must greater than min value!");
            return node.default_single_output_mapping(
                {std::make_shared<opset6::Clamp>(data, min, max)}, {"Out"});
        }

    } // namespace op
} // namespace pdpd