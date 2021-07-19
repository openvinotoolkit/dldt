// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "equal.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs equal(const NodeContext& node)
                {
                    auto data_x = node.get_ng_input("X");
                    auto data_y = node.get_ng_input("Y");
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Equal>(data_x, data_y)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
