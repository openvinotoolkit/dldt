// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_12
            {
                OutputVector einsum(const Node& node);

            } // namespace set_12
        }     // namespace op

    } // namespace onnx_import

} // namespace ngraph
