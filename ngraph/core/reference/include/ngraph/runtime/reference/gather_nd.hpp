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

#include <algorithm>
#include <cassert>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            ///
            /// Implementation find maximum length of *slice* of input *params* which might be
            /// copied to *out* index by index.
            /// +-------+--------------+-------+
            /// | batch | indices[:-1] | slice |
            /// | shape |   shape      | shape |
            /// +-------+--------------+-------+
            ///
            template <typename T, typename U>
            void gather_nd(const T* const params,
                           const U* const indices,
                           T* const out,
                           const Shape& params_shape,
                           const Shape& indices_shape,
                           const Shape& out_shape,
                           const int batch_dims = 0)
            {
                using std::begin;
                using std::end;
                using std::next;
                using std::prev;
                const auto rbegin = [](const Shape& s) { // generic since C++14
                    return s.rbegin();
                };

                const Shape batch_shape(begin(params_shape), next(begin(params_shape), batch_dims));
                const auto batch_size = shape_size(batch_shape);

                // out_shape should have on first dim multiplication of batch number of first
                // dimensions of shape
                assert(!batch_dims || batch_size == out_shape.front());

                // dimensions in params and indices have to be equal on batch dimensions
                assert(std::equal(begin(params_shape),
                                  next(begin(params_shape), batch_dims),
                                  begin(indices_shape)));

                const auto first_slice_index_in_params = batch_dims + indices_shape.back();

                // params_shape should have enough rank to be index by indices
                assert(first_slice_index_in_params <= params_shape.size());

                const auto slice_shape = Shape(
                    next(begin(params_shape), first_slice_index_in_params), end(params_shape));
                const auto slice_size = shape_size(slice_shape);

                const auto indices_offsets = [&] {
                    std::vector<size_t> offsets{slice_size};
                    const auto beg_s = next(rbegin(params_shape), slice_shape.size());
                    const auto end_s = next(beg_s, indices_shape.back() - 1);
                    for (auto s = beg_s; s != end_s; ++s)
                    {
                        const auto o = offsets.back() * *s;
                        offsets.push_back(o);
                    }

                    std::reverse(begin(offsets), end(offsets));
                    return offsets;
                }();

                assert(indices_shape.back() == indices_offsets.size());

                const auto batch_offset = indices_offsets.front() * params_shape[batch_dims];

                const Shape k_1_indecies(next(begin(indices_shape), batch_dims),
                                         prev(end(indices_shape)));

                const Shape k_1_params(next(begin(params_shape), batch_dims),
                                       prev(end(params_shape)));

                const auto number_of_slices_to_copy_in_one_batch = shape_size(k_1_indecies);

                const auto coordinates_size = indices_shape.back();

                for (size_t b = 0; b != batch_size; ++b)
                {
                    const auto input_batch_offset = b * batch_offset;
                    const auto output_batch_offset =
                        b * number_of_slices_to_copy_in_one_batch * slice_size;
                    const auto coordinates_batch_offset =
                        b * number_of_slices_to_copy_in_one_batch * coordinates_size;
                    for (size_t i = 0; i != number_of_slices_to_copy_in_one_batch; ++i)
                    {
                        const auto coordinates =
                            indices + coordinates_batch_offset + i * coordinates_size;

                        size_t mem = input_batch_offset;
                        for (size_t c = 0; c != coordinates_size; ++c)
                        {
                            const auto i_c = coordinates[c];
                            const auto index = i_c < 0 ? k_1_params[c] + i_c : i_c;
                            mem += index * indices_offsets[c];
                        }
                        std::copy(next(params, mem),
                                  next(params, mem + slice_size),
                                  next(out, output_batch_offset + i * slice_size));
                    }
                }
            }

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
