// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/depth_to_space.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void depth_to_space(const char* data,
                                const Shape& data_shape,
                                char* out,
                                const Shape& out_shape,
                                const size_t block_size,
                                const op::DepthToSpace::DepthToSpaceMode mode,
                                const size_t elem_size);
        }
    } // namespace runtime
} // namespace ngraph
