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

#pragma once

#include <string>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        /// Root of all actual ops
        class NGRAPH_API Op : public Node
        {
        protected:
            Op()
                : Node()
            {
            }
            Op(const OutputVector& arguments);
            // To only be removed by OpenVINO
            NGRAPH_DEPRECATED("Use OutputVector constructor instead")
            Op(const NodeVector& nodes)
                : Op(as_output_vector(nodes))
            {
            }
        };
    }
}
