// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <map>
#include <numeric>
#include <vector>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void mean(const T* arg, T* out, const Shape& in_shape, const AxisSet& reduction_axes)
            {
                constexpr bool dont_keep_dims_in_output = false;
                auto out_shape = reduce(in_shape, reduction_axes, dont_keep_dims_in_output);
                std::vector<T> cs(shape_size(out_shape), 0);
                std::fill(out, out + shape_size(out_shape), 0);

                const auto in_strides = row_major_strides(in_shape);
                const auto out_strides = row_major_strides(out_shape);

                CoordinateTransformBasic input_transform(in_shape);
                std::map<size_t, int> index_to_count_map;

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord =
                        reduce(input_coord, reduction_axes, dont_keep_dims_in_output);

                    size_t in_idx = std::inner_product(
                        input_coord.begin(), input_coord.end(), in_strides.begin(), 0);
                    size_t out_idx = std::inner_product(
                        output_coord.begin(), output_coord.end(), out_strides.begin(), 0);

                    T x = arg[in_idx];
                    T& z = out[out_idx];
                    if (index_to_count_map.find(out_idx) == index_to_count_map.end())
                    {
                        index_to_count_map[out_idx] = 1;
                    }
                    else
                    {
                        index_to_count_map[out_idx]++;
                    }

                    if (is_finite(x) && is_finite(z))
                    {
                        T& c = cs[out_idx];
                        T t = z + (x - c);
                        c = (t - z) - (x - c);
                        z = t;
                    }
                    else
                    {
                        z = z + x;
                    }
                }

                for (size_t i = 0; i < shape_size(out_shape); ++i)
                {
                    auto count = index_to_count_map[i];
                    out[i] = out[i] / count;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
