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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph
{
    namespace pass
    {
        class CoreFusion;
    }
}

class NGRAPH_API ngraph::pass::CoreFusion : public ngraph::pass::GraphRewrite
{
public:
    CoreFusion(FusionTypeMask fusions = FusionType::REGULAR_FUSIONS)
        : GraphRewrite()
    {
        if (fusions.is_set(FusionType::REGULAR_FUSIONS))
        {
            construct_relu();
            construct_sigmoid();
            construct_reshape_broadcast();
            construct_reshape_softmax_reshape();
        }
    }
    void construct_relu();
    void construct_sigmoid();
    void construct_reshape_broadcast();
    void construct_reshape_softmax_reshape();
};
