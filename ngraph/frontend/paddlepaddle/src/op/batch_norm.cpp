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

#include "batch_norm.hpp"
#include <ngraph/opsets/opset6.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

namespace pdpd
{
    namespace op
    {
        NamedOutputs batch_norm(const NodeContext& node)
        {
            auto data = node.get_ng_input("X");
            auto gamma = node.get_ng_input("Scale");
            auto beta = node.get_ng_input("Bias");
            auto mean = node.get_ng_input("Mean");
            auto variance = node.get_ng_input("Variance");
            return node.default_single_output_mapping(
                {std::make_shared<opset6::BatchNormInference>(
                    data, gamma, beta, mean, variance, node.get_attribute<float>("epsilon"))},
                {"Y"});
        }

    } // namespace op
} // namespace pdpd