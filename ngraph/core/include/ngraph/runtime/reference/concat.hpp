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

#include <cmath>
#include <cstring>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void concat(const std::vector<const T*>& args,
                        T* out,
                        const std::vector<Shape>& in_shapes,
                        const Shape& out_shape,
                        int64_t concatenation_axis)
            {
                size_t data_size = sizeof(T);
                size_t steps = 1;

                for (int i = 0; i < concatenation_axis; ++i)
                {
                    steps *= out_shape[i];
                }

                size_t out_offset = 0;
                for (size_t step = 0; step < steps; ++step)
                {
                    for (size_t in_index = 0; in_index < args.size(); ++in_index)
                    {
                        size_t size = shape_size(in_shapes[in_index]) / steps;
                        size_t in_offset = step * size;

                        std::memcpy(&out[out_offset], &args[in_index][in_offset], size * data_size);

                        out_offset += size;
                    }
                }
            }
        }
    }
}
