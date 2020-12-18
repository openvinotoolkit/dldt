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

#include <memory>

#include "max_pool.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/max_pool.hpp"
#include "onnx_import/core/null_node.hpp"
#include "onnx_import/utils/pooling_factory.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector max_pool(const Node& node)
                {
                    auto max_pool = pooling::PoolingFactory(node).make_max_pool();
                    if (node.get_outputs_size() > 1)
                    {
                        NGRAPH_WARN << (node) << " Optional output `Indices` is not supported. ";
                        // max_pool.emplace_back(std::make_shared<NullNode>()); // IndexError
                        max_pool.emplace_back(max_pool[0]); // No IndexError
                    }
                    return max_pool;
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
